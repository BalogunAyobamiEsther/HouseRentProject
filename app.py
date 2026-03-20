# ================================
# House Price Prediction Streamlit App
# ================================

import streamlit as st
import pandas as pd
import pickle

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

class MultiColumnLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
        self.encoders = {}

    def fit(self, X, y=None):
        for col in self.columns:
            le = LabelEncoder()
            le.fit(X[col])
            self.encoders[col] = le
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col, le in self.encoders.items():
            X_copy[col] = le.transform(X_copy[col])
        return X_copy

# -------------------------------
# 1. Load the saved pipeline
# -------------------------------
with open("house_rent_model.pkl", "rb") as file:
    pipeline = pickle.load(file)

st.title("🏠 House Price Prediction App")

st.write("""
This app predicts house price based on various property features.
""")

# -------------------------------
# 2. User input function
# -------------------------------
def user_input_features():
    # Numeric inputs
    MSSubClass = st.number_input("MSSubClass", min_value=20, max_value=190, value=60)
    LotFrontage = st.number_input("LotFrontage", min_value=21, max_value=313, value=70)
    LotArea = st.number_input("LotArea", min_value=1300, max_value=215245, value=8000)
    OverallQual = st.number_input("Overall Quality", min_value=1, max_value=10, value=6)
    OverallCond = st.number_input("Overall Condition", min_value=1, max_value=10, value=5)
    YearBuilt = st.number_input("Year Built", min_value=1872, max_value=2021, value=2000)
    YearRemodAdd = st.number_input("Year Remodeled", min_value=1950, max_value=2021, value=2005)
    MasVnrArea = st.number_input("Masonry Veneer Area", min_value=0, max_value=1600, value=100)
    BsmtFinSF1 = st.number_input("Basement Finished SF 1", min_value=0, max_value=5644, value=500)
    BsmtFinSF2 = st.number_input("Basement Finished SF 2", min_value=0, max_value=1474, value=0)
    BsmtUnfSF = st.number_input("Basement Unfinished SF", min_value=0, max_value=2336, value=400)
    TotalBsmtSF = st.number_input("Total Basement SF", min_value=0, max_value=6110, value=900)
    firstFlrSF = st.number_input("1st Floor SF", min_value=334, max_value=4692, value=1000)
    secondFlrSF = st.number_input("2nd Floor SF", min_value=0, max_value=2065, value=500)
    LowQualFinSF = st.number_input("Low Quality Finished SF", min_value=0, max_value=572, value=0)
    GrLivArea = st.number_input("Above Grade Living Area", min_value=334, max_value=5642, value=1500)
    BsmtFullBath = st.number_input("Basement Full Bath", min_value=0, max_value=3, value=1)
    BsmtHalfBath = st.number_input("Basement Half Bath", min_value=0, max_value=2, value=0)
    FullBath = st.number_input("Full Bath", min_value=0, max_value=3, value=2)
    HalfBath = st.number_input("Half Bath", min_value=0, max_value=2, value=1)
    BedroomAbvGr = st.number_input("Bedrooms Above Grade", min_value=0, max_value=8, value=3)
    KitchenAbvGr = st.number_input("Kitchens Above Grade", min_value=0, max_value=3, value=1)
    TotRmsAbvGrd = st.number_input("Total Rooms Above Grade", min_value=2, max_value=12, value=6)
    Fireplaces = st.number_input("Fireplaces", min_value=0, max_value=3, value=1)
    GarageCars = st.number_input("Garage Cars", min_value=0, max_value=4, value=2)
    GarageArea = st.number_input("Garage Area", min_value=0, max_value=1418, value=500)
    WoodDeckSF = st.number_input("Wood Deck SF", min_value=0, max_value=736, value=0)
    OpenPorchSF = st.number_input("Open Porch SF", min_value=0, max_value=547, value=0)
    EnclosedPorch = st.number_input("Enclosed Porch", min_value=0, max_value=552, value=0)
    threeSsnPorch = st.number_input("3-Season Porch", min_value=0, max_value=508, value=0)
    ScreenPorch = st.number_input("Screen Porch", min_value=0, max_value=480, value=0)
    PoolArea = st.number_input("Pool Area", min_value=0, max_value=800, value=0)
    MiscVal = st.number_input("Miscellaneous Value", min_value=0, max_value=15500, value=0)
    MoSold = st.number_input("Month Sold", min_value=1, max_value=12, value=6)
    YrSold = st.number_input("Year Sold", min_value=2006, max_value=2010, value=2008)
    TotalBathrooms = st.number_input("Total Bathrooms", min_value=0, max_value=6, value=3)
    HouseAge = st.number_input("House Age", min_value=0, max_value=150, value=20)
    TotalArea = st.number_input("Total Area", min_value=334, max_value=10000, value=1500)

    # Categorical inputs (replace default values with common categories)
    MSZoning = st.selectbox("MS Zoning", ["RL", "RM", "C (all)", "FV", "RH"])
    Street = st.selectbox("Street", ["Pave", "Grvl"])
    LotShape = st.selectbox("Lot Shape", ["Reg", "IR1", "IR2", "IR3"])
    LandContour = st.selectbox("Land Contour", ["Lvl", "Bnk", "HLS", "Low"])
    Utilities = st.selectbox("Utilities", ["AllPub", "NoSeWa"])
    LotConfig = st.selectbox("Lot Config", ["Inside", "FR2", "Corner", "CulDSac", "FR3"])
    LandSlope = st.selectbox("Land Slope", ["Gtl", "Mod", "Sev"])
    Neighborhood = st.text_input("Neighborhood", "CollgCr")
    Condition1 = st.text_input("Condition1", "Norm")
    Condition2 = st.text_input("Condition2", "Norm")
    BldgType = st.selectbox("BldgType", ["1Fam", "2FmCon", "Duplex", "TwnhsE", "Twnhs"])
    HouseStyle = st.selectbox("HouseStyle", ["1Story", "2Story", "1.5Fin", "SLvl", "SFoyer", "2.5Unf"])
    RoofStyle = st.selectbox("RoofStyle", ["Gable", "Hip", "Gambrel", "Mansard", "Flat"])
    RoofMatl = st.selectbox("RoofMatl", ["CompShg", "Metal", "WdShngl"])
    Exterior1st = st.selectbox("Exterior1st", ["VinylSd", "HdBoard", "MetalSd", "Wd Sdng", "BrkFace"])
    Exterior2nd = st.selectbox("Exterior2nd", ["VinylSd", "HdBoard", "MetalSd", "Wd Sdng", "BrkFace"])
    ExterQual = st.selectbox("ExterQual", ["TA", "Gd", "Ex", "Fa"])
    ExterCond = st.selectbox("ExterCond", ["TA", "Gd", "Ex", "Fa"])
    Foundation = st.selectbox("Foundation", ["PConc", "CBlock", "BrkTil", "Slab", "Stone"])
    BsmtQual = st.selectbox("BsmtQual", ["TA", "Gd", "Ex", "Fa", "NA"])
    BsmtCond = st.selectbox("BsmtCond", ["TA", "Gd", "Ex", "Fa", "NA"])
    BsmtExposure = st.selectbox("BsmtExposure", ["No", "Mn", "Av", "Gd", "NA"])
    BsmtFinType1 = st.selectbox("BsmtFinType1", ["GLQ", "ALQ", "BLQ", "Rec", "LwQ", "Unf", "NA"])
    BsmtFinType2 = st.selectbox("BsmtFinType2", ["GLQ", "ALQ", "BLQ", "Rec", "LwQ", "Unf", "NA"])
    Heating = st.selectbox("Heating", ["GasA", "GasW", "Grav", "Wall", "OthW", "Floor"])
    HeatingQC = st.selectbox("HeatingQC", ["Ex", "Gd", "TA", "Fa", "Po"])
    CentralAir = st.selectbox("CentralAir", ["Y", "N"])
    Electrical = st.selectbox("Electrical", ["SBrkr", "FuseF", "FuseA", "FuseP", "Mix"])
    KitchenQual = st.selectbox("KitchenQual", ["TA", "Gd", "Ex", "Fa"])
    Functional = st.selectbox("Functional", ["Typ", "Min1", "Min2", "Mod", "Maj1", "Maj2", "Sev"])
    GarageType = st.selectbox("GarageType", ["Attchd", "Detchd", "BuiltIn", "CarPort", "Basment", "NA"])
    GarageFinish = st.selectbox("GarageFinish", ["Fin", "RFn", "Unf", "NA"])
    GarageQual = st.selectbox("GarageQual", ["Ex", "Gd", "TA", "Fa", "Po", "NA"])
    GarageCond = st.selectbox("GarageCond", ["Ex", "Gd", "TA", "Fa", "Po", "NA"])
    PavedDrive = st.selectbox("PavedDrive", ["Y", "P", "N"])

    data = {
        'MSSubClass': MSSubClass, 'MSZoning': MSZoning, 'LotFrontage': LotFrontage, 'LotArea': LotArea,
        'Street': Street, 'LotShape': LotShape, 'LandContour': LandContour, 'Utilities': Utilities,
        'LotConfig': LotConfig, 'LandSlope': LandSlope, 'Neighborhood': Neighborhood,
        'Condition1': Condition1, 'Condition2': Condition2, 'BldgType': BldgType, 'HouseStyle': HouseStyle,
        'OverallQual': OverallQual, 'OverallCond': OverallCond, 'YearBuilt': YearBuilt, 'YearRemodAdd': YearRemodAdd,
        'RoofStyle': RoofStyle, 'RoofMatl': RoofMatl, 'Exterior1st': Exterior1st, 'Exterior2nd': Exterior2nd,
        'MasVnrArea': MasVnrArea, 'ExterQual': ExterQual, 'ExterCond': ExterCond, 'Foundation': Foundation,
        'BsmtQual': BsmtQual, 'BsmtCond': BsmtCond, 'BsmtExposure': BsmtExposure, 'BsmtFinType1': BsmtFinType1,
        'BsmtFinSF1': BsmtFinSF1, 'BsmtFinType2': BsmtFinType2, 'BsmtFinSF2': BsmtFinSF2, 'BsmtUnfSF': BsmtUnfSF,
        'TotalBsmtSF': TotalBsmtSF, 'Heating': Heating, 'HeatingQC': HeatingQC, 'CentralAir': CentralAir,
        'Electrical': Electrical, '1stFlrSF': firstFlrSF, '2ndFlrSF': secondFlrSF, 'LowQualFinSF': LowQualFinSF,
        'GrLivArea': GrLivArea, 'BsmtFullBath': BsmtFullBath, 'BsmtHalfBath': BsmtHalfBath, 'FullBath': FullBath,
        'HalfBath': HalfBath, 'BedroomAbvGr': BedroomAbvGr, 'KitchenAbvGr': KitchenAbvGr, 'KitchenQual': KitchenQual,
        'TotRmsAbvGrd': TotRmsAbvGrd, 'Functional': Functional, 'Fireplaces': Fireplaces, 'GarageType': GarageType,
        'GarageYrBlt': YearBuilt, 'GarageFinish': GarageFinish, 'GarageCars': GarageCars, 'GarageArea': GarageArea,
        'GarageQual': GarageQual, 'GarageCond': GarageCond, 'PavedDrive': PavedDrive, 'WoodDeckSF': WoodDeckSF,
        'OpenPorchSF': OpenPorchSF, 'EnclosedPorch': EnclosedPorch, '3SsnPorch': threeSsnPorch, 'ScreenPorch': ScreenPorch,
        'PoolArea': PoolArea, 'MiscVal': MiscVal, 'MoSold': MoSold, 'YrSold': YrSold, 'SaleType': 'WD', 'SaleCondition': 'Normal',
        'TotalBathrooms': TotalBathrooms, 'HouseAge': HouseAge, 'TotalArea': TotalArea
    }

    features = pd.DataFrame([data])
    return features

input_df = user_input_features()

# -------------------------------
# 3. Make prediction
# -------------------------------
if st.button("Predict Price"):
    prediction = pipeline.predict(input_df)
    st.success(f"Estimated Price: ₦{prediction[0]:,.0f}")