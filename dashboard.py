import pandas as pd  # pip install pandas openpyxl
import plotly.express as px  # pip install plotly-express
import streamlit as st  # pip install streamlit
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import shap
import plotly.express as px


# emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="Dashboard Scoring Client", page_icon=":moneybag:", layout="wide")

def main() :
     # ---- MAINPAGE ----
    st.title(":moneybag: Dashboard Scoring Client ")
    st.markdown("##")

    @st.cache
    def load_data():
       
        data=pd.read_csv("sample.csv")
        sample_test=pd.read_csv("sample_test.csv")

        return data,sample_test
    
    @st.cache
    def load_model():
        pickle_in = open('model_log.pkl', 'rb') 
        clf = pickle.load(pickle_in)
        return clf

    @st.cache
    def load_prediction(sample_test, id, clf):
        X=sample_test.iloc[:, :-1]
        score = clf.predict_proba(X[X.index == int(id)])[:,1]
        return score

    @st.cache
    def load_infos_gen(data):
        lst_infos = [data.shape[0],
                     round(data["AMT_INCOME_TOTAL"].mean(), 2),
                     round(data["AMT_CREDIT"].mean(), 2)]

        nb_credits = lst_infos[0]
        rev_moy = lst_infos[1]
        credits_moy = lst_infos[2]

        targets = data.TARGET.value_counts()

        return nb_credits, rev_moy, credits_moy, targets

    @st.cache
    def identite_client(data, id):
        data_client = data[data.index == int(id)]
        return data_client

    @st.cache
    def load_age_population(data):
        data_age = round((data["DAYS_BIRTH"]), 2)
        return data_age

    @st.cache
    def load_income_population(data):
        df_income = pd.DataFrame(data["AMT_INCOME_TOTAL"])
        df_income = df_income.loc[df_income['AMT_INCOME_TOTAL'] < 200000, :]
        return df_income

    
   #Loading data……
    data,sample_test = load_data()
    id_client = data.index.values
    clf = load_model()

   
    #Customer ID selection
    st.sidebar.header("**General Info**")

    #Loading selectbox
    chk_id = st.sidebar.selectbox("Client ID", id_client)

    #Loading general info
    nb_credits, rev_moy, credits_moy, targets = load_infos_gen(data)
    star_rating = ":star:" * int(round(5 ,0))
    
    left_column, middle_column, right_column = st.columns(3)

    with left_column:
        st.subheader("Number of loans in the sample :")
        st.subheader(f"{nb_credits:,}")

    with middle_column:
        st.subheader("Average income (USD) :")
        st.subheader(f"US ${rev_moy} {star_rating}")
    with right_column:
        
        st.subheader("Average loan amount (USD):")
        st.subheader(f"US $ {credits_moy}")

    st.markdown("""---""")



    
    #PieChart
   
    #st.sidebar.markdown("<u>......</u>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(5,5))
    plt.pie(targets, explode=[0, 0.1], labels=['No default', 'Default'], autopct='%1.1f%%', startangle=90)
    fig.set_facecolor("#0083B8")
    st.sidebar.pyplot(fig)
    #######################################
    # HOME PAGE - MAIN CONTENT
    #######################################
    #Display Customer ID from Sidebar
    st.write("Customer ID selection :", chk_id)


    #Customer information display : Customer Gender, Age, Family status, Children, …
    st.header("**Customer information display**")

    if st.checkbox("Show customer information ?"):

        infos_client = identite_client(data, chk_id)
        st.write("**Gender : **", infos_client["CODE_GENDER"].values[0])
        st.write("**Age : **{:.0f} ans".format(int(infos_client["DAYS_BIRTH"])))
        st.write("**Family status : **", infos_client["NAME_FAMILY_STATUS"].values[0])
        st.write("**Number of children : **{:.0f}".format(infos_client["CNT_CHILDREN"].values[0]))

        #Age distribution plot
        data_age = load_age_population(data)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data_age, edgecolor = 'k', color="goldenrod", bins=20)
        ax.axvline(int(infos_client["DAYS_BIRTH"].values / 365), color="green", linestyle='--')
        ax.set(title='Customer age', xlabel='Age(Year)', ylabel='')
        st.pyplot(fig)
    
        
        st.subheader("*Income (USD)*")
        st.write("**Income total : **{:.0f}".format(infos_client["AMT_INCOME_TOTAL"].values[0]))
        st.write("**Credit amount : **{:.0f}".format(infos_client["AMT_CREDIT"].values[0]))
        st.write("**Credit annuities : **{:.0f}".format(infos_client["AMT_ANNUITY"].values[0]))
        st.write("**Amount of property for credit : **{:.0f}".format(infos_client["AMT_GOODS_PRICE"].values[0]))


        #Income distribution plot
        data_income = load_income_population(data)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data_income["AMT_INCOME_TOTAL"], edgecolor = 'k', color="goldenrod", bins=10)
        ax.axvline(int(infos_client["AMT_INCOME_TOTAL"].values[0]), color="green", linestyle='--')
        ax.set(title='Customer income', xlabel='Income (USD)', ylabel='')
        st.pyplot(fig)



        
        

if __name__ == '__main__':
    main()