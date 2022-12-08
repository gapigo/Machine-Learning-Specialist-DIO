(primeiro.vetor1 <- c(1, 3, 5, 9, 10))
(primeiro.vetor2 <- c(1, 4, 5, 8, 22))
(primeiro.vetor3 <- c(primeiro.vetor1, primeiro.vetor2))

length(primeiro.vetor1)

A <- matrix(c(2, 4, 3, 1, 5, 7),nrow = 2,ncol = 3,byrow = TRUE)
B <- matrix(c(2, 5, 3, 4, 5, 12),nrow = 2,ncol = 3,byrow = TRUE)

print(A*B)
