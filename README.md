# litelearn

a python library for building models without fussing
over the nitty gritty details for data munging

once you have a `pandas` dataframe you can create a model 
for your dataset in 3 lines of code:

```python
# load some dataset
import seaborn as sns
dataset = "penguins"
target = "body_mass_g"
df = sns.load_dataset(dataset).dropna(subset=[target])

# just 3 lines of code to create 
# and evaluate a model
import litelearn as ll
model = ll.core_regress_df(df, target)
result = model.get_evaluation() 
```

## installation
`pip install litelearn`