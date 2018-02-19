# Trending Foods

### Plan of Action: 
* Scrape recipe information (title, ingredient list) from [Food52](http://www.food52.com)
* Perform NLP on the text data
* Extract themes and trends from vectorized text


### Load the Recipes:
``` 
import pandas as pd  
df = pd.read_pickle('data/featured_recipes.pkl')  
```

### Plot Ingredient Usage Over Time: 
```
from src.plotters import time_food_plot, yearly_food_plot
```

Pick any ingredient, and see how its use in recipes has changed over time! 
``` 
time_food_plot(df, food='Basil', n_months=1, save=False) 
```

![](https://github.com/elisabeth-berg/trending-food/blob/master/img/basil_popularity.png)

For a given ingredient, plot the monthly frequency of use for each year, along with the average monthly use. 
``` 
yearly_food_plot(df, food='Basil', save=False)
```

![](https://github.com/elisabeth-berg/trending-food/blob/master/img/avg_basil.png)
