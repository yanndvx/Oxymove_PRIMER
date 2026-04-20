# Title:        Linear Mixed Model for Moxy study RPE analysis
# Authors:      L. Borot and Y. Devaux
# Date:         2026-03-01
# Version:      0.1.1
# Description:  Fit a Linear mixed model for RPE data (MOXY project) based on
#               the CR100 Borg scale. 
# Inputs:       ./data/moxy_rpe_table.xlsx in a long form
# Outputs:      
# Dependencies: lme4, lmerTest, car, MuMIn, emmeans, readr, readxl, dplyr,...
#               ggplot2, DHARMa, performance, see, report, broom.mixed
# ------------------------------------------------------------------------------

# Load packages ----
#install.packages(c("lme4", "lmerTest", "car", "MuMIn", "emmeans", "readr", "readxl", 
#                   "dplyr", "ggplot2", "DHARMa", "performance", "see", "report", 
#                   "broom.mixed"))
library(lme4)
library(lmerTest)
library(car)
library(MuMIn)
library(emmeans)
library(readr)
library(readxl)
library(dplyr)
library(ggplot2)
library(DHARMa)
library(performance)
library(see)
library(report)
library(broom.mixed)

# 1-load data ----
# CR100 Borg scale data file (formatted in long form)
data <- read_excel("C:/Program Files/DigiMove/DigiMove/DataAnalysisProject/data/MOXY-bids/phenotype/moxy_rpe_table.xlsx")

# Make factors (e.g. unsure all conditions are considered as a factor for the following steps)
data$condition <-as.factor(data$condition)
data$intensity <-as.factor(data$intensity)
data$trial <-as.factor(data$trial)

# Unsure DV is considered as numerical value
data$rpe<-as.numeric(data$rpe)

# To use if we need to collapse data over trial to avoid pseudo-replication issue

# data <- data %>%
# group_by(subject, condition, intensity) %>%
#summarise(Data = mean(rpe, na.rm = TRUE)) %>%
#ungroup()

# 2-Run model ----

# Start modeling using lmer4 (exemple)
# Should set the random structure for intercept and slope 

# Run full model  <- lmer(Y ~ X1 * X2 + (X1 * X2 | ID), random slope with correlation

# Be careful, risk of singular fit with small amount of data
lmerRPE<- lmer(rpe ~ condition*intensity+trial+ (condition*intensity  | subject) ,control=lmerControl(optimizer="bobyqa",optCtrl=list(maxfun=2e6)),data = data)

# 3-Check Model Assumptions using 'performance' library ----
 
# A. Visual check of all assumptions (Normality, Homoscedasticity, Random Effects, etc.)
# This will generate a multi-panel plot.
check_model(lmerRPE)
 
# B. Specific metric checks (return p-value in the console)
check_collinearity(lmerRPE)  # Check for Multicollinearity (VIF)
check_normality(lmerRPE)     # Shapiro-Wilk test on residuals
check_heteroscedasticity(lmerRPE) # Check for constant variance
 
# if the previous check failed then use glmer with family = Gamma(link = "log") to deal with issues
# e.g. glmerRPE<- glmer(rpe ~ condition*intensity+trial+ (condition*intensity  | subject) ,family = Gamma(link = "log"),control=glmerControl(optimizer="bobyqa",optCtrl=list(maxfun=2e6)),data = data)
 
# C. Check for Singular Fit (common in complex random effects structures like the one we use)
check_singularity(lmerRPE)
 
# if singular fit, run reduced model, random slope without correlation with intercept (force independence)
# e.g. lmerRPE2<- lmer(rpe ~ condition * intensity + trial + (1  | subject) + (0 + condition  | subject) + (0 + intensity  | subject) + (0 + condition:intensity  | subject),control=lmerControl(optimizer="bobyqa",optCtrl=list(maxfun=2e6)),data = data)
# if still singular fit then reduce again 

# 3bis-Model comparison ----
# comparisons of models (using likelihood ratio test), if non significant use parsimonious principle
#anova(lmerRPE,lmerRPE2)
 
# 4-Return main and interaction effects of full  model ----
Anova(lmerRPE, type = "2",test.statistic="Chi")

# give conditionnal (explain variance for full model) and marginal (explain variance for fixed effects) R2
r.squaredGLMM(lmerRPE)
 
# extract post-hoc pairwise comparisons for significant effect with bonferroni correction
test1 = emmeans(lmerRPE, pairwise~ condition*intensity, adjust="bonferroni", type="response")
summary(test1, infer = c(TRUE, TRUE))

# 5-Create standardise report for Oxymove laboratory ----
 
# Generate a comprehensive report of your Mixed Model
model_report <- report(lmerRPE)

# View the output in your console
print(model_report)

# 6- Visualization of Interactions ----

# Extract marginal means for the interaction
em_interaction <- emmeans(lmerRPE, ~ condition * intensity) %>%
  as.data.frame()

# Create the plot
ggplot(em_interaction, aes(x = intensity, y = emmean, color = condition, group = condition)) +
  # Add error bars (Confidence Intervals)
  geom_errorbar(aes(ymin = lower.CL, ymax = upper.CL), width = 0.1, size = 0.8) +
  # Add lines to show the interaction trend
  geom_line(linewidth = 1) +
  # Add points for the means
  geom_point(size = 3) +
  # Formatting for a clean look
  labs(
    title = "Interaction Effect on RPE",
    subtitle = "Marginal means with 95% Confidence Intervals",
    x = "Intensity",
    y = "Estimated Marginal Mean and confidence interval (RPE)",
    color = "Condition"
  ) +
  # Apply white background and remove grids
  theme_bw() + 
  theme(
    panel.grid.major = element_blank(), 
    panel.grid.minor = element_blank(),
    panel.border = element_rect(colour = "black", fill=NA, size=1),
    axis.line = element_line(colour = "black"),
    legend.position = "top"
  )

# Plot the random intercept effect ----
# (to see visually if there is some interest to do that)
# Extract random effects
re_effects <- as.data.frame(ranef(lmerRPE))


# Filter for just the Intercept to see overall subject differences
ggplot(re_effects[re_effects$term == "(Intercept)", ], 
       aes(x = reorder(grp, condval), y = condval)) +
  geom_errorbar(aes(ymin = condval - condsd, ymax = condval + condsd), 
                width = 0, linewidth = 0.6, color = "gray40") +
  geom_point(color = "#2c3e50", size = 2) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  coord_flip() + # Easier to read subject IDs on the Y-axis
  labs(title = "Random Intercepts by Subject",
       subtitle = "Deviations from the global mean RPE",
       x = "Subject ID",
       y = "Conditional standard deviation") +
  theme_bw() +
  theme(panel.grid = element_blank())


# Plot the random slop effect
# Supplement original data with model predictions (including random effects)
augmented_data <- augment(lmerRPE)

ggplot(augmented_data, aes(x = intensity, y = .fitted, color = condition)) +
  # Individual subject lines (Random Slopes)
  geom_line(aes(group = interaction(subject, condition)), 
            alpha = 0.3, linewidth = 0.5) + 
  # Global model trend (Fixed Effect)
  stat_summary(aes(group = condition), fun = mean, geom = "line", 
               linewidth = 1.5, linetype = "solid") +
  labs(title = "Individual Random Slopes vs. Fixed Effects",
       subtitle = "Faint lines = Individual Subjects | Bold lines = Model Average",
       x = "Intensity",
       y = "Predicted RPE") +
  facet_wrap(~condition) +
  theme_bw() +
  theme(panel.grid = element_blank(), legend.position = "none")

