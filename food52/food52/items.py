import scrapy


class RecipeItem(scrapy.Item):
    url = scrapy.Field()
    title = scrapy.Field()
    author = scrapy.Field()
    author_url = scrapy.Field()
    date = scrapy.Field()
    n_comments = scrapy.Field()
    comment_dates = scrapy.Field()
    rating = scrapy.Field()
    n_ratings = scrapy.Field()
    n_faves = scrapy.Field()
    ingredients = scrapy.Field()
    tags = scrapy.Field()
