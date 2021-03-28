! Solve:
!     1. A . eigvec = eigval * S . eigvec
!     2. A . S . eigvec = eigval * eigvec
!     3. S . A . eigvec = eigval * eigvec
! N order real symmetric matrix A
! N order real symmetric positive definite matrix S
! eigval harvests the eigenvalues in ascending order
! if (eigvec) A harvests the eigenvectors normalized under S metric
! S harvests the Cholesky L . L^T decomposition
! Return 0 if normal termination
integer*4 function My_dsygv(A, S, eigval, N, type, eigenvectors)
    integer*4, intent(in)::N
    real*8, dimension(N,N),intent(inout)::A, S
    real*8, dimension(N),intent(out)::eigval
    integer*4, intent(in)::type
    logical, intent(in)::eigenvectors
    real*8, dimension(3*N)::work
    if (eigenvectors) then
        call dsygv(type, 'V','L', N, A, N, S, N, eigval, work, 3*N, My_dsygv)
    else
        call dsygv(type, 'N','L', N, A, N, S, N, eigval, work, 3*N, My_dsygv)
    end if
end function My_dsygv