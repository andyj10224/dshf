/*
 * @BEGIN LICENSE
 *
 * dshf by Psi4 Developer, a plugin to:
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2019 The Psi4 Developers.
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This file is part of Psi4.
 *
 * Psi4 is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Psi4 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along
 * with Psi4; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * @END LICENSE
 */

#include "psi4/psi4-dec.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/liboptions/liboptions.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libfock/jk.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/integral.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/mintshelper.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/vector.h"
#include <memory>

namespace psi{ 
namespace dshf {

class Dshf : public Wavefunction
{
public:
    Dshf(SharedWavefunction ref_wfn, Options& options);
    virtual ~Dshf();

    double compute_energy();
    double compute_electronic_energy();

private:
    Dimension virtpi_;
    void common_init();

    // Max number of SCF iterations
    int maxiter_;
    // Print level
    int print_;
    // Energy convergence
    double e_convergence_;
    // Density convergence
    double d_convergence_;

    // Nuclear Repulsion Energy
    double e_nuc_;

    // Overlap
    SharedMatrix S_;
    // One Electron Hamiltonian
    SharedMatrix H_;
    // Orthogonalizer
    SharedMatrix X_;
    // Fock Matrix
    SharedMatrix F_;
    // Transformed Fock Matrix
    SharedMatrix Ft_;
    // MO Coefficients
    SharedMatrix C_;
    // Occupied Part of MO coefficients
    SharedMatrix Co_;
    // Density Matrix
    SharedMatrix D_;

    // Integral Factory
    std::shared_ptr<IntegralFactory> integral_;
    // ERI Integral Object
    TwoBodyAOInt* eri_;

    // Libfock JK Object
    // std::shared_ptr<JK> jk_;

};

Dshf::Dshf(SharedWavefunction ref_wfn, Options& options)
    : Wavefunction(options)
{
    // Shallow copy ref_wfn data into this wavefunction
    shallow_copy(ref_wfn);

    // Read in options from input file
    print_ = options.get_int("PRINT");
    maxiter_ = options.get_int("HF_MAXITER");
    e_convergence_ = options.get_double("E_CONVERGENCE");
    d_convergence_ = options.get_double("D_CONVERGENCE");

    common_init();
}

Dshf::~Dshf()
{
}

void Dshf::common_init()
{
    // nsopi_, frzcpi_, etc are Dimension objects for symmetry orbitals
    // These are copied from ref_wfn when we call for shallow_copy
    virtpi_ = nsopi_ - frzcpi_ - frzvpi_ - doccpi_;

    outfile->Printf("The wavefunction has the following dimensions:\n");
    nsopi_.print();
    frzcpi_.print();
    doccpi_.print();
    virtpi_.print();
    frzvpi_.print();

    // Print out molecule
    molecule_->print();
    if (print_ > 1) {
        basisset_->print_detail();
    }

    // Nuclear repulsion energy without a field
    e_nuc_ = molecule_->nuclear_repulsion_energy({0, 0, 0});
    outfile->Printf("\n    Nuclear repulsion energy: %16.8f\n\n", e_nuc_);

    // Make a MintsHelper object
    MintsHelper mints(basisset_);

    // Obtain the basic integrals
    S_ = mints.ao_overlap();
    H_ = mints.ao_kinetic();
    H_->add(mints.ao_potential());

    if (print_ > 3) {
        S_->print();
        H_->print();
    }

    integral_ = mints.integral();
    eri_ = integral_->eri();

    // jk_ = JK::build_JK(basisset_, std::shared_ptr<BasisSet>(), options_);
    // jk_->set_memory(Process::environment.get_memory() * 0.8);
    // jk_->initialize();
    // jk_->print_header();

}

double Dshf::compute_electronic_energy() {
    SharedMatrix HplusF = H_->clone();
    HplusF->add(F_);
    return D_->vector_dot(HplusF);
}

double Dshf::compute_energy()
{
    /* Your code goes here. */
    // Dimension zero(1);

    int nbf = S_->nrow();
    int ndocc = nalpha();

    // Allocate Matrices
    X_ = std::make_shared<Matrix>("S^(-1/2)", nbf, nbf);
    F_ = std::make_shared<Matrix>("Fock Matrix", nbf, nbf);
    Ft_ = std::make_shared<Matrix>("Transformed Fock Matrix", nbf, nbf);
    C_ = std::make_shared<Matrix>("MO Coefficients", nbf, nbf);
    Co_ = std::make_shared<Matrix>("Occupied MO coefficients", nbf, ndocc);
    D_ = std::make_shared<Matrix>("Density Matrix", nbf, nbf);

    // Allocate Temp Matrices
    auto Temp1 = std::make_shared<Matrix>("Temp1", nbf, nbf);
    auto Temp2 = std::make_shared<Matrix>("Temp2", nbf, nbf);
    auto FDS = std::make_shared<Matrix>("FDS", nbf, nbf);
    auto SDF = std::make_shared<Matrix>("SDF", nbf, nbf);
    auto Evecs = std::make_shared<Matrix>("Eigenvectors", nbf, nbf);
    auto Evals = std::make_shared<Vector>("Eigenvalues", nbf);

    // Obtain the orbital vector from JK object and insert in our occupied MO coefficient matrix
    // std::vector<SharedMatrix> &Cl = jk_->C_left();
    // Cl.clear();
    // Cl.push_back(Co_);

    // Form the X_ matrix (S^-1/2)
    X_->copy(S_);
    X_->power(-0.5);


    F_->copy(H_);
    Ft_->transform(F_, X_);
    Ft_->diagonalize(Evecs, Evals, ascending);

    // C := alpha * transa(A) * transb(B) + beta * C
    C_->gemm(false, false, 1.0, X_, Evecs, 0.0);
    
    for (int i = 0; i < nbf; i++) {
        for (int j = 0; j < ndocc; j++){
            Co_->set(i, j, C_->get(i, j));
        }
    }

    // Form the density matrix
    D_->gemm(false, true, 1.0, Co_, Co_, 0.0);

    if (print_ > 1) {
        outfile->Printf("MO Coefficients and density from core Hamiltonian guess:\n");
        C_->print();
        Co_->print();
        D_->print();
    }

    int iter = 1;
    bool converged = false;
    double e_old = 0.0;
    double e_new = e_nuc_ + compute_electronic_energy();

    outfile->Printf("    *=======================================================*\n");
    outfile->Printf("    * Iter       Energy          delta E     || gradient || *\n");
    outfile->Printf("    *-------------------------------------------------------*\n");

    while (!converged && iter <= maxiter_) {
        e_old = e_new;

        // Add the core Hamiltonian to the Fock operator
        F_->copy(H_);

        // jk_->compute();

        // const std::vector<SharedMatrix> &J = jk_->J();
        // const std::vector<SharedMatrix> &K = jk_->K();

        SharedMatrix Jmat = std::make_shared<Matrix>("J Matrix", nbf, nbf);
        SharedMatrix Kmat = std::make_shared<Matrix>("K Matrix", nbf, nbf);

        Jmat->set(0.0);
        Kmat->set(0.0);

        int nshell = basisset_->nshell();

        int n_start, m_start, k_start, l_start;
        int num_n, num_m, num_k, num_l;

        for (int N = 0; N < nshell; N++) {
            n_start = basisset_->shell(N).function_index();
            num_n = basisset_->shell(N).nfunction();
            for (int M = 0; M < nshell; M++) {
                m_start = basisset_->shell(M).function_index();
                num_m = basisset_->shell(M).nfunction();
                for (int K = 0; K < nshell; K++) {
                    k_start = basisset_->shell(K).function_index();
                    num_k = basisset_->shell(K).nfunction();
                    for (int L = 0; L < nshell; L++) {
                        l_start = basisset_->shell(L).function_index();
                        num_l = basisset_->shell(L).nfunction();
                        
                        eri_->compute_shell(N, M, K, L);
                        const double* buffer = eri_->buffer();

                        for (int n = n_start; n < n_start + num_n; n++) {
                            for (int m = m_start; m < m_start + num_m; m++) {
                                for (int k = k_start; k < k_start + num_k; k++) {
                                    for (int l = l_start; l < l_start + num_l; l++) {

                                        int dn = n - n_start;
                                        int dm = m - m_start;
                                        int dk = k - k_start;
                                        int dl = l - l_start;

                                        int nmkl = buffer[dn * num_m * num_k * num_l + dm * num_k * num_l + dk * num_l + dl];
                                        int nkml = buffer[dn * num_k * num_m * num_l + dk * num_m * num_l + dm * num_l + dl];
                                        
                                        Jmat->set(n, m, Jmat->get(n, m) + D_->get(k, l) * nmkl);
                                        Kmat->set(n, m, Kmat->get(n, m) + D_->get(k, l) * nkml);

                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // F = 2J - K
        Jmat->scale(2.0);
        F_->add(Jmat);
        F_->subtract(Kmat);

        // Compute the energy
        e_new = e_nuc_ + compute_electronic_energy();
        double dE = e_new - e_old;

        // Compute the orbital gradient, FDS - SDF
        FDS = linalg::triplet(F_, D_, S_);
        SDF = linalg::triplet(S_, D_, F_);

        Temp1->copy(FDS);
        Temp1->subtract(SDF);

        // Density RMS
        double dRMS = Temp1->rms();

        converged = (fabs(dE) < e_convergence_) && (dRMS < d_convergence_);

        outfile->Printf("     * %3d %20.14f     %9.2e     %9.2e    *\n", iter, e_new, dE, dRMS);

        if (converged)
            break;

        // Transform the Fock operator and diagonalize it
        Ft_->transform(F_, X_);
        Ft_->diagonalize(Evecs, Evals, ascending);

        // Form the orbitals
        C_->gemm(false, false, 1.0, X_, Evecs, 0.0);

        // Update our occupied orbitals and density
        // Co_->copy(C_->get_block({zero, nsopi_}, {zero, doccpi_}));

        for (int i = 0; i < nbf; i++) {
            for (int j = 0; j < ndocc; j++){
                Co_->set(i, j, C_->get(i, j));
            }
        }
        D_->gemm(false, true, 1.0, Co_, Co_, 0.0);

        iter += 1;
    }
    outfile->Printf("    *=============================================================*\n");

    // Save the energy to the wavefunction
    energy_ = e_new;

    return e_new;
}

extern "C" PSI_API
int read_options(std::string name, Options& options)
{
    if (name == "DSHF"|| options.read_globals()) {
        /*- The amount of information printed to the output file -*/
        options.add_int("PRINT", 1);

        /*- How tightly to converge the energy -*/
        options.add_double("E_CONVERGENCE", 1.0E-10);

        /*- How tightly to converge the density -*/
        options.add_double("D_CONVERGENCE", 1.0E-6);

        /*- How many iterations allowed -*/
        options.add_int("HF_MAXITER", 50);
    }

    return true;
}

extern "C" PSI_API
SharedWavefunction dshf(SharedWavefunction ref_wfn, Options& options)
{

    // Note that if this function was integrated into Psi4 we would not be using P::e.wavefunction
    // Instead everything would be explicitly passed
    auto wfn = std::make_shared<Dshf>(ref_wfn, options);
    wfn->compute_energy();

    return wfn;
}

}} // End namespaces

