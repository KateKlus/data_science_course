import pandas as pd
import numpy as np

authors = pd.DataFrame({
    "author_id": [1, 2, 3],
    "author_name": ['Тургенев', 'Чехов', 'Островский']
})

books = pd.DataFrame({
    "author_id": [1, 1, 1, 2, 2, 3, 3],
    "book_title": ['Отцы и дети', 'Рудин', 'Дворянское гнездо', 'Толстый и тонкий', 'Дама с собачкой', 'Гроза', 'Таланты и поклонники'],
    "price": [450, 300, 350, 500, 450, 370, 290]
})

authors_price = pd.merge(authors, books, on='author_id', how='inner')
print(authors_price)

top5 = authors_price.sort_values(by="price", ascending=False)[0:5]
print(top5)

group_by = authors_price.groupby("author_name")

authors_stat = pd.DataFrame({
    "min_price": group_by["price"].min(),
    "max_price": group_by["price"].max(),
    "mean_price": group_by["price"].mean()
})

print(authors_stat)

cover = ['твердая', 'мягкая', 'мягкая', 'твердая', 'твердая', 'мягкая', 'мягкая']
authors_price.loc[authors_price["book_title"].notnull(), "cover"] = cover
print(authors_price)

book_info = pd.pivot_table(data=authors_price, index="author_name", fill_value=0, aggfunc={'price': np.sum},
                           columns=['cover'])
print(book_info)

book_info.to_pickle("book_info.pkl")
book_info2 = pd.read_pickle("book_info.pkl")
print(book_info.equals(book_info2))