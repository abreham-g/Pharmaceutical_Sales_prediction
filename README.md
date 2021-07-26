# Pharmaceutical_Sales_prediction
# Data and Features
### You can use the following questions as a guide during your analysis. It is important to come
### up with more questions to explore. This is part of our expectation for an excellent analysis.
    ● Check for seasonality in both training and test sets - are the seasons similar between
    these two groups?● Check & compare sales behavior before, during and after holidays
    ● Find out any seasonal (Christmas, Easter etc) purchase behaviours,
    ● What can you say about the correlation between sales and number of customers?
    ● How does promo affect sales? Are the promos attracting more customers? How does
    it affect already existing customers?
    ● Could the promos be deployed in more effective ways? Which stores should promos
    be deployed in?
    ● Trends of customer behavior during store open and closing times
    ● Which stores are opened on all weekdays? How does that affect their sales on
    weekends?
    ● Check how the assortment type affects sales
    ● How does the distance to the next competitor affect sales? What if the store and its
    competitors all happen to be in city centres, does the distance matter in that case?
    ● How does the opening or reopening of new competitors affect stores? Check for
    stores with NA as competitor distance but later on has values for competitor distance

# Data fields
Most of the fields are self-explanatory. The following are descriptions for those that aren't.
### Id  
    - an Id that represents a (Store, Date) duple within the test setStore - a unique Id for each store
### Sales 
    - the turnover for any given day (this is what you are predicting)
### Customers 
    - the number of customers on a given day
### Open 
    - an indicator for whether the store was open: 0 = closed, 1 = open
### StateHoliday 
    - indicates a state holiday. Normally all stores, with few exceptions, are
    closed on state holidays. Note that all schools are closed on public holidays and
    weekends. a = public holiday, b = Easter holiday, c = Christmas, 0 = None
    SchoolHoliday - indicates if the (Store, Date) was affected by the closure of public
    schools
### StoreType 
    - differentiates between 4 different store models: a, b, c, d
### Assortment 
    - describes an assortment level: a = basic, b = extra, c = extended. Read more
    about assortment here
### CompetitionDistance 
    - distance in meters to the nearest competitor store
### CompetitionOpenSince[Month/Year] 
    - gives the approximate year and month of the
    time the nearest competitor was opened
### Promo 
    - indicates whether a store is running a promo on that day
### Promo2 
    - Promo2 is a continuing and consecutive promotion for some stores: 0 = store is
not participating, 1 = store is participating
### Promo2Since[Year/Week] 
    - describes the year and calendar week when the store
    started participating in Promo2
### PromoInterval 
    - describes the consecutive intervals Promo2 is started, naming the
    months the promotion is started anew. E.g. "Feb,May,Aug,Nov" means each round starts in
    February, May, August, November of any given year for that store
