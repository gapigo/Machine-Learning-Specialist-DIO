# Cria um vetor concatenado
x <- c(12, 7, 3, 4.2, 18, 2, 54, -21, 8, -5, 7)

result.mean <- mean(x)
print(result.mean)

result.median <- median(x)
print(result.median)

Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

result.mode <- Mode(x)
print(result.mode)
