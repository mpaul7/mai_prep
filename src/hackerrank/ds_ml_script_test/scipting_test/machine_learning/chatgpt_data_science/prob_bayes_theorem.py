# Example: Disease test
# P(Disease) = 0.01, P(Pos|Disease)=0.95, P(Pos|No Disease)=0.05
P_D = 0.01
P_Pos_given_D = 0.95
P_Pos_given_NoD = 0.05
P_NoD = 1 - P_D

# Bayes theorem: P(Disease|Positive)
P_D_given_Pos = (P_Pos_given_D * P_D) / (P_Pos_given_D * P_D + P_Pos_given_NoD * P_NoD)
print("Probability of disease given positive test:", P_D_given_Pos)
