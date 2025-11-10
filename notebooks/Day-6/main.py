warnings. fllterwarnings("ignore")
#Load
print("Loading data...")
train = pd.read_csv("train.csv') 
test = pd.read_csv("test.csv)
I
*BreakthroughFeature Engineering
def create_breakthrough_ features(df, pca_model-None,scaler-None, fit_transformers-True):
features • [f'Component(1)_fraction' for 1 in range(1, 6)]
features += [f"Component(1)_Property(J)' for 1 in range(1, 6) for 1 in range(2, 11)]
•Enhanced interaction features with non-linear transformations for 1 in range(1,6): for 1 in range(2,11):
df[f"frac(1)_prop(3)"]= df[f"Component(4)_fraction*]*df[fComponent(i)_Property(J)]
df[f"frac(1)_prop(1)_sqrt"]- df[fComponent(1)_fraction']*np.sqrt(np.abs(df[f"Component(1)_Property(1)'1))
df[f"frac(1)_prop(1)_1og']- df[f"Component(1)_fraction"]*np.10g(np.abs(df[F"Component(1)Property(J)])+1)
df[f°frac(4)_prop(J)_square']-df[fComponent(1)_fraction']"(df[f"Component(1)Property(J)"|**2)
features,extend([f"frac(1)_prop(J),f'frac(1)_prop(J)_sqrt', f'frac(1)_prop(J)_1og',f°frac(1)_prop(J)square)
•Advanced weighted features with multiple aggregation methods for j in range(1, 11):
prop_cols • [f'Component(1)_Property(J)"for 1 in range(1, 6)] frac_cols • [f"Component(1)_fraction" for 1 in range(2,6)]
*Multiple weighted aggregations df[fweighted_mean_prop(J)"] - sum(
df[f"Component(1)_fraction']"df[f"Component(1)_Property(J)"]for 1 in range(2,6)
mean • df[fweighted_mean_prop())"]
df[fweighted_var_prop(J)]- sum(
df[f"Component(1)_fraction']"(df[f"Component(1)_Property(J)"]-mean)**2 for 1 in range(2, 6)
*Harmonic mean (important for fuel