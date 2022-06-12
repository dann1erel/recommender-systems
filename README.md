# Рекомендательные системы
### Альт.экзамен ДМиТИ-2022
#### _Участники:_
* Волосевич Артем [@github/Miryz21](https://github.com/Miryz21)
* Кашуба Данил [@github/dann1erel](https://github.com/dann1erel)
* Шлом Илья [@github/ilya-shlom](https://github.com/ilya-shlom)

_(гр.1310)_

## Что такое рекомендательные системы?

**Рекомендательные системы** – комплекс программных решений, направленных
на предсказание предпочтений пользователя. Объектом рекомендательной
системы может стать что угодно – будь то фильмы, сериалы, музыка, книги,
статьи и тексты в интернете, товары в интернет-магазине и так далее.
В настоящее время рекомендательные системы широко применяются
на большинстве крупных интернет-сайтах.

Существуют разные алгоритмы для подбора рекомендаций. Среди них
выделяют два основных – _коллаборативная фильтрация_ и
_фильтрация на основе содержания_. Именно их реализацию вы
сможете найти в данном репозитории.

С более подробной информацией о рекомендательных системах можно ознакомиться [в нашей презентации.](https://docs.google.com/presentation/d/1MrewFc5sMTSZkc3D9ytYA0VA-MH6rlKM7BjHZ5ssg64/edit#slide=id.g13041fd2381_0_5)

##  Инструкция по использованию

**Файл collaborative-system.py** - Пример работы коллаборативной системы с заранее заданными данными и выводом всех промежуточных данных;

**Файл content-based.py** - Пример работы фильтрации на основе содержания - через консоль вводится строка текста и через 20-30 секунд выводится результат (промежуточные данные не даны в связи с тем, что они слишком большие и не дают наглядного представления работы программы.

**Файл config.py** - Конфигурация основных программ - прописаны названия файлов, с которыми происходит работа. Его запускать не надо!

_Для запуска программ необходимо подключить приведенные ниже библиотеки, скопировать себе данные из репозитория и запустить через консоль/IDE одну из требуемых программ._

**Библиотеки, которые необходимо подключить:**
* SpaCy
* NumPy
* Pandas
* stop-words

**Про директорию testing_data**
* **ratings-small.csv** - Файл для коллаборативной фильтрации;
* **df_text_eng_big.csv** - оригинальный датасет для фильтрации на основе содержания. Файл слишком большой для обработки на обычном компьютере и нужен лишь для проверки системы на данных, которые не попадают на обработку программе, поэтому программе на обработку можно скормить один из двух файлов:
+ **df_text_eng_new.csv (рекомендуется)** - 4000 строки с соотношением успешных проектов к неуспешным 5/3. При тестированиях было выявлено, что при таком соотношении и количестве файлов программа максимально выигрышна в соотношении скорость работы/попадение в правильный результат;
+ **df_text_eng.csv** - Обрезанный до 10000 строк оригинальный файл.
