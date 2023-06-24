# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 23:58:16 2023

@author: navee
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle

import streamlit.components.v1 as components

from nbconvert import HTMLExporter
import nbformat
from PIL import Image




# Load the model
loaded_model = pickle.load(open('final_model.sav', 'rb'))
# Load the hybrid model components
xgb_model2 = pickle.load(open('xgb_model2.sav', 'rb'))
lgb_model2 = pickle.load(open('lgb_model2.sav', 'rb'))
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

    menu = ["Home", "Predict", "Model Development", "Ydata_Overview","About Me"]
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
        with open('model_development.ipynb', 'r') as f:
            notebook = nbformat.read(f, as_version=4)

    # Convert the Jupyter notebook to HTML
        html_exporter = HTMLExporter()
        html_exporter.template_name = 'classic'
        (body, _) = html_exporter.from_notebook_node(notebook)
        

    # Display the HTML in Streamlit
        components.html(body,width=1000, height=1000, scrolling=True)
    elif choice == "Ydata_Overview":
        with open('profile_report.html', 'r') as f:
            html_string = f.read()
        components.html(html_string, height = 800, scrolling=True)


    elif choice == "About Me":
        st.title('About Me')

        col1, col2 = st.columns(2)

        with col1:
            st.write("""
            ## Naveen Kumar Kadampally
         
            ## Professional Background

           My career journey started as a **Civil Engineer** at RG Constructions where I honed my skills in construction management, contract management, and AutoCAD. My role here was multidimensional, and I was responsible for everything from supervising construction workers, ensuring site safety, to material quantity management. My stint at RG Constructions provided me with a strong foundation and a broad perspective of problem-solving, which was the cornerstone for my subsequent career transitions.

           
        """)

        with col2:
            image = Image.open("img.jpg")
            st.image(image, use_column_width=True)

        
        st.write("""
         I then made a bold career switch into the technology domain as a **Software Test Analyst** at Unified Softech. Intrigued by the role of data in decision-making and the potential it had to revolutionize industries, I navigated my way into data analysis and data science.

        My transformation into a **Data Analyst** was an exciting journey. After acquiring the IBM Data Science Professional Certification, I began leveraging my knowledge in various data analysis tools and Python libraries to derive valuable insights from complex datasets. I played a pivotal role in developing targeted marketing strategies and enhancing customer experience by managing customer complaints more efficiently.

        Building on my data analysis experience, I am currently diving deeper into the realm of **Artificial Intelligence (AI) and Machine Learning**. I am exploring these technologies to build advanced models that can further streamline decision-making and efficiency in businesses.

        ## Skills

        My technical skills range from **AutoCAD** and **Revit** in the field of Civil Engineering to **Python**, **SQL**, **Tableau**, and **PowerBI** in Data Analysis. My knowledge of **Machine Learning algorithms** and **Data Visualization tools** has allowed me to create intuitive dashboards and powerful predictive models.

        On the soft skills front, I pride myself on my strong **communication skills** and **project management** abilities. These skills have been instrumental in leading teams, managing projects, and effectively coordinating with stakeholders.

        ## Projects

        I have been involved in several exciting projects throughout my career journey. My key projects include a **Customer Segmentation Project** at Unified Softech where I used Machine Learning to segment customers based on their purchasing history. The outcome was a more personalized customer experience and improved customer retention rates.

        I also embarked on a project focused on **Predictive Analysis of Sales Data**. This project involved creating a model that could accurately predict future sales based on historical data, helping the company to strategically plan their production and inventory management.

        In a freelance capacity, I worked with a construction company to develop a **Machine Learning model for Green Concrete Strength Prediction**. The model made accurate predictions about the strength of the green concrete based on the composition of recycled materials. This innovative approach promoted sustainability in the construction industry.

        Now, as I venture into AI and Deep Learning, I'm excited to work on more advanced projects that integrate these technologies. My current projects are aimed at exploring new frontiers in AI and I am thrilled about the potential outcomes.
        """)
                 
                 
    
        st.write("""
        ## Certifications

        I have pursued several certifications to expand my knowledge base and validate my skills, including:

        - **IBM Data Science Professional Certification**: This certification provided a solid grounding in data science methodologies, and tools including Python, SQL, and data visualization.

        - **ISTQB Certified Tester**: This internationally recognized certification has honed my understanding of software testing techniques and principles. It played a pivotal role during my tenure as a Software Test Analyst at Unified Softech.

        - **Microsoft Power BI Certification**: This certification confirmed my expertise in using PowerBI for data analysis and visualization, enabling me to create effective and insightful dashboards.

        - **AutoCAD Certified User**: This certification validated my proficiency in AutoCAD software, which I utilized extensively during my tenure as a Civil Engineer.

        ## Learning Journey and Future Goals

        Transitioning from Civil Engineering to Software Testing, and then Data Science has been an incredible journey of learning and growth. This journey has taught me that technology's potential is boundless, and the more we learn, the more we can harness its power.

        As I venture further into the world of AI and Machine Learning, I aim to create more sophisticated models and solutions that can enhance decision-making and efficiency in businesses.

        I am also eager to contribute to the growing field of AI ethics, as I believe that responsible AI usage is crucial for future technological advancements. In line with this, I am working towards obtaining a certification in AI Ethics in the coming year.

        ## Personal Interests

        I have a profound affinity for the outdoors. I am captivated by nature and relish every opportunity to be outside, whether that's on sun-drenched beaches or in the verdant expanses of a forest. I find these moments to be incredibly refreshing and grounding, a perfect respite from the hustle and bustle of the tech world. 

        Mountains, in particular, hold a special allure for me. I love trekking through mountainous terrains and setting up camp under the stars. There's something incredibly serene about the stillness of the night, punctuated only by the gentle crackle of a campfire and the laughter and stories shared around it. 
    
        Lakes are another favorite escape of mine, especially those nestled in the heart of mountains. I enjoy boating and fishing, finding these activities to be both exciting and soothing. There's a unique sense of contentment I experience when I am casting a line out into the tranquil water, surrounded by the majesty of mountain peaks.
    
        Beyond my love for outdoor activities, I am an internet enthusiast. I spend a considerable amount of time surfing the web, staying updated on emerging trends and technologies. This curiosity extends to my professional life, as it helps me keep a pulse on the rapidly evolving tech landscape.
    
        My family plays a pivotal role in my life. I treasure the time spent watching movies with my children and exploring the outdoors with them. I believe these experiences foster their curiosity, teach them resilience, and create priceless memories.
    
        My interests, whether it's my love for nature or technology, or my devotion to family, shape who I am. They provide me with relaxation, continual learning, and cherished experiences – elements that I consider vital in life.

        ## Contact Information

        I'm always open to discussing data science, technology, and potential collaborations. You can reach me via email at [naveenkadampally@outlook.com](mailto:naveenkadampally@outlook.com).

        ### Social Media Profiles
        [![Instagram](https://img.shields.io/badge/-Instagram-black?style=flat-square&logo=instagram)](https://www.instagram.com/naveen.ka202/)
        [![LinkedIn](https://img.shields.io/badge/-LinkedIn-black?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/naveen-kumar-kadampally/)
        [![GitHub](https://img.shields.io/badge/-GitHub-black?style=flat-square&logo=github)](https://github.com/nani2357)
    """)
    

if __name__ == "__main__":
    main()
