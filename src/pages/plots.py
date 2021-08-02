import streamlit as st
import awesome_streamlit as ast
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def write():
    with st.spinner("Loading Plots ..."):
        st.title('Data Visualisation')

        # read the datasets
        na_value=['',' ','nan','Nan','NaN','na', '<Na>']
        train = pd.read_csv('src/pages/train.csv', na_values=na_value)
        store = pd.read_csv('src/pages/store.csv', na_values=na_value)
        full_train = pd.merge(left = train, right = store, how = 'inner', left_on = 'Store', right_on = 'Store')
        #st.sidebar.title("Gallery")
        st.sidebar.subheader("Choose Feature to plot")
        plot = st.sidebar.selectbox("feature", ( "Correlation",'Promotions', 'State Holiday', 'PromoIntervals', 'Assortment','Competition',"Seasonality",))

        if plot == 'Competition':
            st.subheader("Competition Distance")
            st.set_option('deprecation.showPyplotGlobalUse', False)
            full_train['Decile_rank'] = pd.qcut(full_train['CompetitionDistance'], 5, labels = False) 
            new_df = full_train[['Decile_rank', 'Sales']]
            a = new_df.groupby('Decile_rank').mean()
            labels = a.index.to_list()
            sizes = a.Sales.to_list()
            fig = plt.figure(figsize =(10, 7)) 
            colors = ['gold', 'yellowgreen', 'purple', 'lightcoral', 'lightskyblue']
            explode = (0.1, 0.03, 0.03, 0.03, 0.03)  # explode 1st slice

            plt.pie(sizes, explode=explode, labels=labels, colors=colors, shadow=True, autopct='%.2f', startangle=140)
            plt.title('A piechart indicating mean sales in the 5 CompetitioDIstance decile classes')
            #st.pyplot()
            full_train['Decile_rank'] = pd.qcut(full_train['CompetitionDistance'], 5, labels = False) 
            new_df = full_train[['Decile_rank', 'Customers']]
            a = new_df.groupby('Decile_rank').mean()
            labels = a.index.to_list()
            sizes = a.Customers.to_list()
            fig = plt.figure(figsize =(10, 7)) 
            colors = ['gold', 'yellowgreen', 'purple', 'lightcoral', 'lightskyblue']
            explode = (0.1, 0.03, 0.03, 0.03, 0.03)  # explode 1st slice

            # Plot
            plt.pie(sizes, explode=explode, labels=labels, colors=colors, shadow=True, autopct='%.2f', startangle=140)
            plt.title('A piechart indicating mean number of customers in the 5 CompetitioDistance decile classes')
            st.pyplot(fig)

        if plot == 'Seasonality':
            st.subheader("Daily, Weekly and Monthly Averaged Sales Seasonality Plot")
            st.set_option('deprecation.showPyplotGlobalUse', False)
            time_data = full_train[['Date', 'Sales']]
            time_data['datetime'] = pd.to_datetime(time_data['Date'])
            time_data = time_data.set_index('datetime')
            time_data = time_data.drop(['Date'], axis = 1)

            daily_time_data = time_data.Sales.resample('D').mean() 
            plt.figure(figsize = (12,5))
            plt.figure(figsize = (12,5))
            plt.title('Seasonality plot averaged daily')
            daily_time_data.plot()
            plt.grid() 
            st.pyplot()  
            weekly_time_data = time_data.Sales.resample('W').mean() 
            plt.figure(figsize = (12,5))
            plt.title('Seasonality plot averaged weekly')
            plt.ylabel('average sales')
            weekly_time_data.plot()
            plt.grid()
            st.pyplot()

            #monthly
            monthly_time_data = time_data.Sales.resample('M').mean() 
            plt.figure(figsize = (15,7))
            plt.title('Seasonality plot averaged monthly')
            plt.ylabel('average sales')
            monthly_time_data.plot()


            plt.grid()
            st.pyplot()
        if plot == 'Correlation':            
            st.subheader("Linear Relationships between the Sales and customers")
            st.set_option('deprecation.showPyplotGlobalUse', False)
            def correlation_map(f_data, f_feature, f_number):
                f_most_correlated = f_data.corr().nlargest(f_number,f_feature)[f_feature].index
                f_correlation = f_data[f_most_correlated].corr()
                
                f_mask = np.zeros_like(f_correlation)
                f_mask[np.triu_indices_from(f_mask)] = True
                with sns.axes_style("white"):
                    f_fig, f_ax = plt.subplots(figsize=(8, 6))
                    f_ax = sns.heatmap(f_correlation, mask=f_mask, vmin=0, vmax=1, square=True,
                                    annot=True, annot_kws={"size": 10}, cmap="BuPu")

                plt.show()
                plt.figure(figsize=(10,9))
                sns.heatmap(f_data.corr(), linewidths=0.1, vmax=1.0, 
                    square=True, cmap=plt.cm.RdBu, linecolor='white', annot=True)

            print('top 6 features with highest correlation with sales')
            correlation_map(full_train, 'Sales', 6)
            st.pyplot()
            st.write("""
            Sales and Customers have a high correlation. 
            This is because sales are directly dependent on the number of customers.
            """)


        # PromoIntervals plots
        if plot == 'PromoIntervals':
            st.subheader("Promotion Intervals")
            st.set_option('deprecation.showPyplotGlobalUse', False)
            flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
            sns.countplot(x='PromoInterval', data=full_train, palette = flatui).set_title('PromoInterval value counts')
            st.pyplot()
            fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

            sns.barplot(x='PromoInterval', y='Sales', data=full_train, ax=axis1, palette = flatui).set_title('sales across different promo intervals')
            sns.barplot(x='PromoInterval', y='Customers', data=full_train, ax=axis2, palette = flatui).set_title('customers across different promo intervals')
            st.pyplot()


        # Promotions plots
        if plot == 'Promotions':
            flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
            st.subheader("Countplot and Barplots indicating Promotions and Sales and customers across the stores")
            sns.countplot(x='Promo', data=full_train, palette = flatui).set_title('Promo counts')
            st.pyplot()
            fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))
            sns.barplot(x='Promo', y='Sales', data=full_train, ax=axis1, palette = flatui).set_title('sales across different Promo')
            sns.barplot(x='Promo', y='Customers', data=full_train, ax=axis2, palette = flatui).set_title('customers across different Promo')
            st.pyplot()



        # State Holiday plots
        if plot == 'State Holiday':
            st.subheader("Sales During State Holidays and Ordinary Days")
            st.set_option('deprecation.showPyplotGlobalUse', False)
            full_train["StateHoliday"].loc[full_train["StateHoliday"] == 0] = "0"
            sns.countplot(x='StateHoliday', data=full_train, palette = 'Paired').set_title('State holidays value counts')
            st.pyplot()
            fig, (axis1,axis2) = plt.subplots(1,2,figsize=(12,4))
            sns.barplot(x='StateHoliday', y='Sales', data=full_train, ax=axis1, palette = 'Paired').set_title('comparison of sales during StateHolidays and ordinary days')
             
            mask = (full_train["StateHoliday"] != "0") & (full_train["Sales"] > 0)
            sns.barplot(x='StateHoliday', y='Sales', data=full_train[mask], ax=axis2, palette = 'Paired').set_title('sales during Stateholidays')
            st.pyplot()

            fig, (axis1,axis2) = plt.subplots(1,2,figsize=(12,4))
            sns.barplot(x='StateHoliday', y='Customers', data=full_train, ax=axis1, palette = 'Paired').set_title('comparison of customers during StateHolidays and ordinary days')
            # holidays only
            mask = (full_train["StateHoliday"] != "0") & (full_train["Customers"] > 0)
            sns.barplot(x='StateHoliday', y='Customers', data=full_train[mask], ax=axis2, palette = 'Paired').set_title('customers during Stateholidays')
            st.pyplot()
        if plot == 'Assortment':
            st.subheader("Sales across different assortment types")
            st.set_option('deprecation.showPyplotGlobalUse', False)
            sns.countplot(x='Assortment', data=full_train, order=['a','b','c'], palette = 'husl').set_title('assortment types counts')
            st.pyplot()
            fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))
            sns.barplot(x='Assortment', y='Sales', data=full_train, order=['a','b','c'], palette = 'husl', ax = axis1).set_title('sales across different assortment types')
            sns.barplot(x='Assortment', y='Customers', data=full_train, order=['a','b','c'], ax=axis2, palette = 'husl').set_title('Number of customers across different assortment types')
            st.pyplot()
            st.write("""
            The store counts in the 3 assortment classes. Basic(a) and extended(c) are the most populated.
            The sales volumes across the 3 classes. Despite  the extra(b) class having the least number of stores, it has the highest volume of sales.
            """)

        # Store Type plots
