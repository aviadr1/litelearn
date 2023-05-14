# litelearn

a python library for building models without fussing
over the nitty gritty details for data munging


## installation
`pip install litelearn`

## usage

once you have a `pandas` dataframe you can create a model 
for your dataset in 3 lines of code:

### Regression
```python
# load some dataset
import seaborn as sns

dataset = "penguins"
target = "body_mass_g"
df = sns.load_dataset(dataset).dropna(subset=[target])

# just 3 lines of code to create and evaluate a model
import litelearn as ll

model = ll.regress_df(df, target)
model.display_evaluation() 
```

### Classification
```python
# load some dataset
import seaborn as sns

dataset = "penguins"
target = "species"
df = sns.load_dataset(dataset).dropna(subset=[target])

# just 3 lines of code to create and evaluate a model
import litelearn as ll

model = ll.classify_df(df, target)
model.display_evaluation()
```

### Prediction
prediction is easy too, it will work on any data that resembles the training data.
dtypes don't have to match, and you can even have extra columns in your prediction data.
missing values and unknown categories will be imputed with the training data's values.

```python
df = ...  # load some dataframe
split = int(len(df) * 0.8)
train_df, val_df = df[:split], df[split:] 
model = ...  # build some model
pred = model.predict(val)  # predict on unseen data
```


## features
+ does all the data munging for you, including missing data, categorical data handling
+ uses the robust [catboost](https://catboost.ai/) library for gradient boosting, which is known for generating
  high quality models with little tuning
+ supports [shap](https://github.com/slundberg/shap) for explainability.
  call `model.display_shap()` or `model.get_shap()` to get the shap values for your model
+ supports sklearn's [permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html)
  call `model.display_permutation_importance()` or `model.get_permutation_importance()` 
  to get feature importances that are biased towards the model's performance on test data.
+ supports easy pickling: to save your model simply call `model.dump("path/to/model.pkl")`
  and to load your model call `model.load("path/to/model.pkl")`
+ for regression models, you can call `model.display_residuals()` to see the residuals of your model
+ it also supports segmeents for your data using the `model.set_segments()` method.
  this will create a new column in your dataframe called `segment` which you can use to
  group your data. this is useful for seeing how your model performs on different segments of your data.
  

## interactive streamlit demo
To showcase the library, see [here](https://aviadr1-litelearn-streamlit-main-bkhh48.streamlit.app/) for an interactive 
streamlit web app showing the ability of the library to handle many datasets with ease, 
including uploading your own dataset. 

![streamlit demo](https://awesomescreenshot.s3.amazonaws.com/image/1882885/39822988-68e9b3563f6895357c1eea07f99ee1c4.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJSCJQ2NM3XLFPVKA%2F20230514%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230514T215550Z&X-Amz-Expires=28800&X-Amz-SignedHeaders=host&X-Amz-Signature=c96a347d40fb645271d846cac0f612895be5ed1d0eaeadf04bfb21ba991d44ae)

the code for this demo is on [github.com/aviadr1/litelearn-streamlit](https://github.com/aviadr1/litelearn-streamlit)
