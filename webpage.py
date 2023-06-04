# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 23:58:16 2023

@author: navee
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gdown
import streamlit.components.v1 as components
import os
from nbconvert import HTMLExporter
import nbformat


print("Current working directory:", os.getcwd())
print("Files in working directory:", os.listdir())

# Load the model
loaded_model = pickle.load(open(r'C:\Users\navee\Green_Concrete_Strength_predication_model\final_model.sav', 'rb'))
# Load the hybrid model components
xgb_model2 = pickle.load(open(r'C:\Users\navee\Green_Concrete_Strength_predication_model\xgb_model2.sav', 'rb'))
lgb_model2 = pickle.load(open(r'C:\Users\navee\Green_Concrete_Strength_predication_model\lgb_model2.sav', 'rb'))
# Create a function to make predictions
def make_prediction(cement, age, blast_furnace_slag, coarse_aggregate, fine_aggregate, fly_ash, superplasticizer, water):
    new_data = pd.DataFrame([
        [cement, age, blast_furnace_slag, coarse_aggregate, fine_aggregate, fly_ash, superplasticizer, water]],
        columns=['cement', 'age', 'blast_furnace_slag', 'coarse_aggregate', 'fine_aggregate', 'fly_ash', 'superplasticizer', 'water'])
    prediction = loaded_model.predict(new_data)
    return prediction

# Creating a function for the hybrid model
def make_hybrid_prediction(xgb_model, lgb_model, cement, age, blast_furnace_slag, coarse_aggregate, fine_aggregate, fly_ash, superplasticizer, water):
    new_data = pd.DataFrame([
        [cement, age, blast_furnace_slag, coarse_aggregate, fine_aggregate, fly_ash, superplasticizer, water]],
        columns=['cement', 'age', 'blast_furnace_slag', 'coarse_aggregate', 'fine_aggregate', 'fly_ash', 'superplasticizer', 'water'])
    prediction = (xgb_model.predict(new_data) + lgb_model.predict(new_data)) / 2
    return prediction

def show_html_file(file_name):
    with open(file_name, 'r') as f:
        html_string = f.read()
    st.markdown(html_string, unsafe_allow_html=True)
    
# Create the Streamlit app
def main():
    st.title("Predictive Modeling for Concrete Strength Using Recycled and Traditional Materials")

    menu = ["Home", "Predict", "Model Development", "Ydata_Overview"]
    choice = st.sidebar.radio("Menu", menu)

    if choice == "Home":
        st.write("Disclaimer: This project serves as a prototype and is a simplified representation of an actual client project. Due to confidentiality agreements and privacy considerations, the original client project and associated data cannot be shared or disclosed. Any data used in this project is synthetic or anonymized, and any resemblance to actual events or persons is purely coincidental.")
        st.subheader("Green Concrete: A Sustainable Revolution in Construction")
        st.write("Green Concrete refers to the innovative use of concrete made from \
                 recycled materials. As we step into an era of environmental \
                     consciousness, it's gaining popularity for its potential to\
                         reduce the carbon footprint of the construction industry.")
        st.subheader("Benefits of Green Concrete")
        st.write("Unlike traditional concrete, Green Concrete uses recycled materials,\
                 minimizing the use of non-renewable resources and cutting down on waste.\
                     Its unique composition contributes to its strength, durability,\
                         and workability, making it a promising solution for a more \
                             sustainable future in construction.")
        st.subheader("Essential Ingredients of Green Concrete")
        st.write(" My predictive model uses data from eight key\
                 ingredients to estimate the compressive strength of Green Concrete.")
        st.markdown("<h4 style='text-align: left; color: black;'>1. Cement: The Backbone of Strength</h4>", unsafe_allow_html=True)
        st.write("As the primary ingredient, cement provides the essential backbone of strength and stability in Green Concrete.")

        st.markdown("<h4 style='text-align: left; color: black;'>2. Blast Furnace Slag: Enhancing Durability</h4>", unsafe_allow_html=True)
        st.write("A byproduct from the iron-making process, Blast Furnace Slag is used to enhance the durability and resistance of the blocks, contributing to a longer lifespan and reducing the need for replacements.")
        
        st.markdown("<h4 style='text-align: left; color: black;'>3. Fly Ash: Adding Strength and Workability</h4>", unsafe_allow_html=True)
        st.write("This byproduct of burning coal contributes to the strength and workability of Green Concrete. By utilizing this waste product, we not only improve the quality of the blocks but also decrease the environmental impact of coal production.")
        
        st.markdown("<h4 style='text-align: left; color: black;'>4. Water: Binding the Elements</h4>", unsafe_allow_html=True)
        st.write("Water plays a pivotal role in the chemical reaction that binds all the components together, forming a solid, cohesive structure.")
        
        st.markdown("<h4 style='text-align: left; color: black;'>5. Superplasticizer: Streamlining Workability</h4>", unsafe_allow_html=True)
        st.write("Superplasticizer is an additive that improves the workability and flow of the Green Concrete mixture. It ensures the mix is easy to shape and mold, making the construction process more efficient.")
        
        st.markdown("<h4 style='text-align: left; color: black;'>6. Coarse Aggregate: Adding Reinforcement</h4>", unsafe_allow_html=True)
        st.write("Coarse aggregate, usually consisting of crushed stone or gravel, is added for reinforcement and stability, playing a significant role in the strength and durability of the blocks.")
        
        st.markdown("<h4 style='text-align: left; color: black;'>7. Fine Aggregate: Filling the Gaps</h4>", unsafe_allow_html=True)
        st.write("Fine aggregate, such as sand, fills in the gaps between the coarse aggregates, resulting in a smoother, more cohesive mixture. This fine tuning enhances the overall structural integrity of the blocks.")
        
        st.markdown("<h4 style='text-align: left; color: black;'>8. Age: The Maturing Factor</h4>", unsafe_allow_html=True)
        st.write("The age of the concrete is a critical determinant of its strength and durability. The longer it cures, the stronger it gets. This factor is taken into account in this predictive model to provide accurate estimates of Green Concrete strength.")
        st.subheader("The Problem with Traditional Concrete")
        
        st.write("Traditional concrete, while a popular building material, has its share of environmental drawbacks. The production of concrete consumes a substantial amount of non-renewable resources and energy. Moreover, it significantly contributes to CO2 emissions globally, aggravating the ongoing climate crisis. Traditional concrete production also generates considerable waste, putting added pressure on our already strained waste management systems.")
        st.subheader("The Solution: Green Concrete")
        st.write("Green Concrete emerges as a sustainable alternative, addressing these environmental challenges head-on. It incorporates recycled and waste materials into the mix, significantly reducing reliance on non-renewable resources. The innovative use of byproducts, like fly ash and blast furnace slag, also cuts down on industrial waste. With a lower carbon footprint, Green Concrete paves the way for a sustainable future in construction, without compromising on strength and durability.")
        st.subheader("My Prediction Model")
        st.write("My platform currently leverages two advanced machine learning models, XGBoost Regression and a hybrid model combining XGBoost and LightGBM. Both models have proven their efficiency and accuracy, achieving an impressive 93% accuracy on the test dataset, surpassing industry benchmarks. Moreover, these models maintain an equally remarkable accuracy with 10-fold cross-validation, further attesting to their robustness.")
        st.write("The hybrid model, a blend of XGBoost and LightGBM, stands out for its lower mean and median error. It also exhibits a distribution of error where near 80% of the predictions fall within an acceptable range of error, highlighting its superior reliability despite the limited data. ")
        st.write("These models have been further refined through meticulous hyperparameter tuning, resulting in an even more accurate and reliable model performance. I haveve implemented both models in a live environment to test which model provides the most accurate predictions under real-world conditions. This on-the-ground testing and optimization further underscore my commitment to delivering the most accurate and reliable concrete strength predictions.")
        st.write("However, please note that the predictions provided by these models are estimations. The actual concrete strength may vary within a range of ±5 MPa due to other influencing factors that aren't considered in the models. Therefore, the predictions should be utilized as guiding measures rather than absolute values.")
        st.write("For an in-depth overview of the model development process, please visit 'Model Development' section.")
        st.subheader("Why Use Predictor?")
        st.write("Testing different mixtures of concrete for optimal strength is a costly and time-consuming endeavor. This predictive model offers a solution to these challenges. By leveraging my model, industry professionals can predict the compressive strength of Green Concrete mixtures before testing, saving substantial time and resources in the process. This model also facilitates informed decision-making, helping design teams test mixtures that are likely to yield the most promising results. In this way,this predictor tool empowers industry professionals to streamline their operations and contribute to a sustainable future in construction.")
        
        
        
        
        
        
    elif choice == "Predict":
        st.subheader("Predict the Concrete Strength")
        cement = st.number_input("Cement (Kg/m³)", min_value=0.0, max_value=1000.0, value=0.0, step=0.1)
        age = st.slider("Age (1 to 365 days)",min_value=1, max_value=365,step=1)
        blast_furnace_slag = st.number_input("Blast Furnace Slag (Kg/m³)",min_value=0.0, max_value=1000.0, value=0.0, step=0.1)
        coarse_aggregate = st.number_input("Coarse Aggregate (Kg/m³))",min_value=0.0, max_value=1000.0 ,value=0.0, step=0.1)
        fine_aggregate = st.number_input("Fine Aggregate (Kg/m³))",min_value=0.0, max_value=1000.0, value=0.0, step=0.1)
        fly_ash = st.number_input("Fly Ash (Kg in M^3)", min_value=0.0, max_value=1000.0,value=0.0, step=0.1)
        superplasticizer = st.number_input("Superplasticizer (Kg/m³)",min_value=0.0, max_value=1000.0 ,value=0.0, step=0.1)
        water = st.number_input("Water (Kg/m³)", min_value=0.0, max_value=1000.0,value=0.0, step=0.1)

        # Select model
        model_choice = st.selectbox('Choose a model', ('Model-1 XGB', 'Hybrid Model XGB & LGB'))
        if cement == 0 and blast_furnace_slag == 0 and coarse_aggregate == 0 and fine_aggregate == 0 and fly_ash == 0 and superplasticizer == 0 and water == 0:
            if st.button("Predict"):
                st.write(f"Predicted Concrete Strength: 0 MPa")
        
        else:
            if model_choice == 'Model-1 XGB':
                
        
                if st.button("Predict"):
            
                    result = make_prediction(cement, age, blast_furnace_slag, coarse_aggregate, fine_aggregate, fly_ash, superplasticizer, water)
                    st.write(f"Predicted Concrete Strength: {result[0]} MPa")
            elif model_choice == 'Hybrid Model XGB & LGB':
                if st.button("Predict"):
                    result = make_hybrid_prediction(xgb_model2, lgb_model2, cement, age, blast_furnace_slag, coarse_aggregate, fine_aggregate, fly_ash, superplasticizer, water)
                    st.write(f"Predicted Concrete Strength: {result[0]} MPa")
                     
        st.write("Please note that the predictions generated by the models are estimations, and the actual concrete strength may vary within a range of ±5 MPa. The predictions should be used as a guide, but other factors may also influence the concrete strength that aren't considered in this model. ")
        
        
        
    elif choice == "Model Development":
        st.markdown("<h2 style='text-align: left; color: black;'> Step-by-Step: Building and Optimizing Regression Models Filling the Gaps</h4>", unsafe_allow_html=True)
        
        # Read the Jupyter notebook file
        with open(r'C:\Users\navee\Green_Concrete_Strength_predication_model\model_development.ipynb', 'r') as f:
            notebook = nbformat.read(f, as_version=4)

    # Convert the Jupyter notebook to HTML
        html_exporter = HTMLExporter()
        html_exporter.template_name = 'classic'
        (body, _) = html_exporter.from_notebook_node(notebook)
        

    # Display the HTML in Streamlit
        components.html(body,width=1000, height=1000, scrolling=True)
    elif choice == "Ydata_Overview":
        url = 'https://drive.google.com/uc?id=1TTjeEu5DQ3S38E6-Zoqn3AIBsoxOTQ5R'
        output = 'Profile_report.html'
        gdown.download(url, output, quiet=False)

        with open('Profile_report.html', 'r') as f:
            html_string = f.read()
        components.html(html_string, height = 800, scrolling=True)

if __name__ == "__main__":
    main()