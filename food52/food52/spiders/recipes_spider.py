import scrapy
from items import RecipeItem

class RecipesSpider(scrapy.Spider):
    name = "recipes"
    start_urls = ['https://food52.com/sitemap/recipes']


    def parse(self, response):
        for url in response.css('.content-listing a::attr(href)').extract():
            yield scrapy.Request(url, callback=self.parse_recipe)

    def parse_recipe(self, response):
        title = response.css(
            '.article-header-title::text').extract_first().strip()
        header = response.css('.article-header-meta__item')
        author = header.css('a::text').extract_first()
        author_url = header.css('a::attr(href)').extract_first()
        date = header[1].css('::text').extract_first()
        n_comments = header[2].css('span::text').extract_first()
        comment_dates = response.css(
            '.comment .comment__byline .datetime::text').extract()
        rating = ''
        n_ratings = ''
        n_faves = response.css('.button__label::text').extract_first()
        ingredients = [x.strip() for x in response.css(
            '.recipe-list-item-name::text').extract()]
        ingredients = ' '.join([food for food in ingredients if food !=''])
        tags = response.css('.recipe-tag-landing-pages a::text').extract()
        recipe = RecipeItem(
            title = title,
            author = author,
            author_url = author_url,
            date = date,
            n_comments = n_comments,
            comment_dates = comment_dates,
            rating = rating,
            n_ratings = n_ratings,
            n_faves = n_faves,
            ingredients = ingredients,
            tags = tags,
        )
        yield recipe