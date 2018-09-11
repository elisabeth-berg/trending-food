import scrapy
from items import RecipeItem

class RecipesSpider(scrapy.Spider):
    name = "recipes"
    start_urls = ['https://food52.com/sitemap/recipes']


    def parse(self, response):
        """
        On the first page, extract the total number of pages,
        then send each sitemap page to parse_page.
        """
    #    last_page = response.css('.pagination a::text').extract()[-2]
        last_page = 3
        self.logger.info('Found {} pages'.format(last_page))
        for page in range(int(last_page)):
            next_url = 'https://food52.com/sitemap/recipes?page={}'.format(page)
            yield scrapy.Request(
                next_url,
                callback = self.parse_page,
                meta = {'page' : page + 1})

    def parse_page(self, response):
        """
        Extract each recipe on the page, then send the recipe url to
        parse_recipe.
        """
        self.logger.info('Parsing page {}'.format(response.meta['page']))
        for url in response.css('.content-listing a::attr(href)').extract():
            yield scrapy.Request(
                url,
                callback = self.parse_recipe)

    def parse_recipe(self, response):
        url = response.url
        title = response.css(
            '.article-header-title::text').extract_first().strip()
        header = response.css('.article-header-meta__item')
        author = header.css('a::text').extract_first()
        author_url = header.css('a::attr(href)').extract_first()
        date = header[1].css('::text').extract_first()
        n_comments = header[2].css('span::text').extract_first()
        comment_dates = response.css(
            '.comment .comment__byline .datetime::text').extract()
        # can't for the life of me figure out how to get this rating info
        rating = ''
        n_ratings = ''
        n_faves = response.css('.button__label::text').extract_first()
        ingredients = [x.strip() for x in response.css(
            '.recipe-list-item-name::text').extract()]
        ingredients = ' '.join([food for food in ingredients if food !=''])
        tags = response.css('.recipe-tag-landing-pages a::text').extract()
        recipe = RecipeItem(
            url = url,
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
