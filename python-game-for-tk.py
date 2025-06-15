import tkinter as tk
from tkinter import messagebox
import random

# Класс для хранения информации о репозитории
class Repository:
    def __init__(self, name, issues):
        self.name = name
        self.issues = issues  # Список ошибок в репозитории

# Класс для хранения информации об ошибке
class Issue:
    def __init__(self, description, code, correct_fix, wrong_fixes):
        self.description = description  # Описание ошибки
        self.code = code  # Код с ошибкой
        self.correct_fix = correct_fix  # Правильное исправление
        self.wrong_fixes = wrong_fixes  # Неправильные исправления

# Основной класс игры
class PythonJuniorGame:
    def __init__(self, root):
        self.root = root
        self.root.title("Python Junior: Исправь ошибки!")
        self.root.geometry("600x650")

        # Счет игрока
        self.score = 0
        self.level = 1  # Уровень сложности
        self.progress = 0  # Прогресс прохождения

        # Создаем репозитории с ошибками
        self.repositories = [
            Repository("awesome-project", [
                Issue(
                    "Синтаксическая ошибка в функции",
                    "def add(a, b)\n    return a + b",
                    ["def add(a, b):\n    return a + b"],
                    ["def add(a, b)\n    return a + b", "def add(a, b):\n    return a - b"]
                ),
                Issue(
                    "Ошибка в цикле",
                    "for i in range(10)\n    print(i)",
                    ["for i in range(10):\n    print(i)"],
                    ["for i in range(10)\n    print(i)", "for i in range(10):\n    print(i + 1)"]
                )
            ]),
            Repository("data-science-tools", [
                Issue(
                    "Ошибка в импорте",
                    "import numppy as np",
                    ["import numpy as np"],
                    ["import numppy as np", "import numpy as n"]
                ),
                Issue(
                    "Ошибка в условии",
                    "if x = 5:\n    print(x)",
                    ["if x == 5:\n    print(x)"],
                    ["if x = 5:\n    print(x)", "if x == 5\n    print(x)"]
                )
            ])
        ]

        # Добавляем дополнительные вопросы
        self.repositories.extend([
            Repository(f"project-{i}", [issue]) for i, issue in enumerate(questions, start=3)
        ])

        # Перемешиваем репозитории и задачи внутри них
        random.shuffle(self.repositories)  # Случайная сортировка репозиториев
        for repo in self.repositories:
            random.shuffle(repo.issues)  # Случайная сортировка задач в каждом репозитории

        # Текущий репозиторий и ошибка
        self.current_repo_index = 0
        self.current_issue_index = 0

        # Элементы интерфейса
        self.label_repo = tk.Label(root, text="", font=("Arial", 14))
        self.label_repo.pack(pady=10)

        self.label_issue = tk.Label(root, text="", font=("Arial", 12), wraplength=550)
        self.label_issue.pack(pady=10)

        self.label_code = tk.Label(root, text="", font=("Courier", 12), justify="left")
        self.label_code.pack(pady=10)

        self.buttons = []
        for i in range(3):
            button = tk.Button(root, text="", font=("Arial", 12), width=50, command=lambda i=i: self.check_answer(i))
            button.pack(pady=5)
            self.buttons.append(button)

        self.label_score = tk.Label(root, text=f"Счет: {self.score}", font=("Arial", 14))
        self.label_score.pack(pady=10)

        self.label_level = tk.Label(root, text=f"Уровень: {self.level}", font=("Arial", 14))
        self.label_level.pack(pady=10)

        self.label_progress = tk.Label(root, text=f"Прогресс: {self.progress}%", font=("Arial", 14))
        self.label_progress.pack(pady=10)

        # Начинаем игру
        self.load_repository()

    # Загрузка репозитория
    def load_repository(self):
        if self.current_repo_index >= len(self.repositories):
            self.final_evaluation()
            return

        repo = self.repositories[self.current_repo_index]
        self.label_repo.config(text=f"Репозиторий: {repo.name}")
        self.load_issue()

    # Загрузка ошибки
    def load_issue(self):
        repo = self.repositories[self.current_repo_index]
        if self.current_issue_index >= len(repo.issues):
            self.current_repo_index += 1
            self.current_issue_index = 0
            self.load_repository()
            return

        issue = repo.issues[self.current_issue_index]
        self.label_issue.config(text=f"Ошибка: {issue.description}")
        self.label_code.config(text=f"Код с ошибкой:\n{issue.code}")

        # Создаем варианты ответов
        options = issue.correct_fix + issue.wrong_fixes
        random.shuffle(options)  # Перемешиваем варианты ответов
        for i, button in enumerate(self.buttons):
            button.config(text=options[i])

    # Проверка ответа
    def check_answer(self, button_index):
        repo = self.repositories[self.current_repo_index]
        issue = repo.issues[self.current_issue_index]
        selected_option = self.buttons[button_index].cget("text")

        if selected_option in issue.correct_fix:
            self.score += 10
            messagebox.showinfo("Правильно!", "Ошибка исправлена! +10 очков.")
        else:
            self.score -= 5
            messagebox.showerror("Неправильно!", "Ошибка не исправлена. -5 очков.")

        # Обновляем прогресс
        self.progress = int((self.current_repo_index + 1) / len(self.repositories) * 100)
        self.label_progress.config(text=f"Прогресс: {self.progress}%")

        # Обновляем уровень
        if self.score >= self.level * 100:
            self.level += 1
            self.label_level.config(text=f"Уровень: {self.level}")

        self.label_score.config(text=f"Счет: {self.score}")
        self.current_issue_index += 1
        self.load_issue()

    # Финальная оценка
    def final_evaluation(self):
        if self.score >= 500:
            messagebox.showinfo("Поздравляем!", f"Вы Senior! Ваш счет: {self.score}")
        elif self.score >= 300:
            messagebox.showinfo("Поздравляем!", f"Вы Middle! Ваш счет: {self.score}")
        else:
            messagebox.showinfo("Поздравляем!", f"Вы Junior! Ваш счет: {self.score}")
        self.root.quit()


# Вопросы для игры
questions = [
    # Синтаксические ошибки
    Issue("Синтаксическая ошибка в функции", "def add(a, b)\n    return a + b", ["def add(a, b):\n    return a + b"], ["def add(a, b)\n    return a + b", "def add(a, b):\n    return a - b"]),
    Issue("Синтаксическая ошибка в цикле", "for i in range(10)\n    print(i)", ["for i in range(10):\n    print(i)"], ["for i in range(10)\n    print(i)", "for i in range(10):\n    print(i + 1)"]),
    Issue("Синтаксическая ошибка в условии", "if x = 5:\n    print(x)", ["if x == 5:\n    print(x)"], ["if x = 5:\n    print(x)", "if x == 5\n    print(x)"]),
    Issue("Синтаксическая ошибка в списке", "my_list = [1, 2, 3", ["my_list = [1, 2, 3]"], ["my_list = [1, 2, 3", "my_list = (1, 2, 3)"]),
    Issue("Синтаксическая ошибка в словаре", "my_dict = {1: 'one', 2: 'two'", ["my_dict = {1: 'one', 2: 'two'}"], ["my_dict = {1: 'one', 2: 'two'", "my_dict = [1: 'one', 2: 'two']"]),

    # Ошибки в импорте
    Issue("Ошибка в импорте библиотеки", "import numppy as np", ["import numpy as np"], ["import numppy as np", "import numpy as n"]),
    Issue("Ошибка в импорте модуля", "from math import sqroot", ["from math import sqrt"], ["from math import sqroot", "from math import square"]),
    Issue("Ошибка в импорте класса", "from datetime import dateime", ["from datetime import datetime"], ["from datetime import dateime", "from datetime import date"]),

    # Ошибки в работе с циклами
    Issue("Ошибка в цикле for", "for i in range(10):\n    print(i", ["for i in range(10):\n    print(i)"], ["for i in range(10):\n    print(i", "for i in range(10)\n    print(i)"]),
    Issue("Ошибка в цикле while", "while x < 10:\n    print(x)\n    x += 1", ["while x < 10:\n    print(x)\n    x += 1"], ["while x < 10\n    print(x)\n    x += 1", "while x < 10:\n    print(x)\n    x =+ 1"]),

    # Ошибки в условиях
    Issue("Ошибка в условии if", "if x = 5:\n    print(x)", ["if x == 5:\n    print(x)"], ["if x = 5:\n    print(x)", "if x == 5\n    print(x)"]),
    Issue("Ошибка в условии elif", "if x > 10:\n    print('Больше')\nelif x = 5:\n    print('Равно')", ["if x > 10:\n    print('Больше')\nelif x == 5:\n    print('Равно')"], ["if x > 10:\n    print('Больше')\nelif x = 5:\n    print('Равно')", "if x > 10:\n    print('Больше')\nelif x == 5\n    print('Равно')"]),

    # Ошибки в работе со строками
    Issue("Ошибка в конкатенации строк", "print('Hello' + 5)", ["print('Hello' + str(5))"], ["print('Hello' + 5)", "print('Hello' + '5')"]),
    Issue("Ошибка в форматировании строки", "print(f'Результат: {result}')", ["print(f'Результат: {result}')"], ["print('Результат: {result}')", "print(f'Результат: result')"]),

    # Ошибки в работе с функциями
    Issue("Ошибка в определении функции", "def multiply(a, b):\nreturn a * b", ["def multiply(a, b):\n    return a * b"], ["def multiply(a, b):\nreturn a * b", "def multiply(a, b)\n    return a * b"]),
    Issue("Ошибка в вызове функции", "print(add(2, 3)", ["print(add(2, 3))"], ["print(add(2, 3)", "print(add 2, 3)"]),

    # Ошибки в работе с классами
    Issue("Ошибка в определении класса", "class MyClass:\n    def __init__(self, x):\n    self.x = x", ["class MyClass:\n    def __init__(self, x):\n        self.x = x"], ["class MyClass:\n    def __init__(self, x):\n    self.x = x", "class MyClass:\n    def __init__(self, x)\n        self.x = x"]),
    Issue("Ошибка в создании объекта", "obj = MyClass(5)", ["obj = MyClass(5)"], ["obj = MyClass(5", "obj = MyClass 5)"]),

    # Ошибки в работе с исключениями
    Issue("Ошибка в блоке try-except", "try:\n    x = 1 / 0\nexcept ZeroDivisionError:\n    print('Ошибка')", ["try:\n    x = 1 / 0\nexcept ZeroDivisionError:\n    print('Ошибка')"], ["try:\n    x = 1 / 0\nexcept ZeroDivisionError\n    print('Ошибка')", "try:\n    x = 1 / 0\nexcept:\n    print('Ошибка')"]),
    Issue("Ошибка в блоке finally", "try:\n    x = 1 / 0\nexcept ZeroDivisionError:\n    print('Ошибка')\nfinally\n    print('Завершено')", ["try:\n    x = 1 / 0\nexcept ZeroDivisionError:\n    print('Ошибка')\nfinally:\n    print('Завершено')"], ["try:\n    x = 1 / 0\nexcept ZeroDivisionError:\n    print('Ошибка')\nfinally\n    print('Завершено')", "try:\n    x = 1 / 0\nexcept ZeroDivisionError:\n    print('Ошибка')\nfinally: print('Завершено')"]),

    # Ошибки в работе с файлами
    Issue("Ошибка при открытии файла", "file = open('test.txt', 'r')", ["file = open('test.txt', 'r')"], ["file = open('test.txt', 'r'", "file = open('test.txt', 'w')"]),
    Issue("Ошибка при чтении файла", "file.read()", ["file.read()"], ["file.read", "file.read())"]),

    # Ошибки в работе с модулями
    Issue("Ошибка в использовании модуля math", "import math\nprint(math.sqrt(-1))", ["import math\nprint(math.sqrt(1))"], ["import math\nprint(math.sqrt(-1))", "import math\nprint(math.square(1))"]),
    Issue("Ошибка в использовании модуля random", "import random\nprint(random.randint(1, 10))", ["import random\nprint(random.randint(1, 10))"], ["import random\nprint(random.randint(1, 10)", "import random\nprint(random.rand(1, 10))"]),

    # Ошибки в работе с библиотеками
    Issue("Ошибка в использовании библиотеки numpy", "import numpy as np\nprint(np.array([1, 2, 3]))", ["import numpy as np\nprint(np.array([1, 2, 3]))"], ["import numpy as np\nprint(np.array([1, 2, 3]", "import numpy as np\nprint(np.arr([1, 2, 3]))"]),
    Issue("Ошибка в использовании библиотеки pandas", "import pandas as pd\ndf = pd.DataFrame([1, 2, 3])", ["import pandas as pd\ndf = pd.DataFrame([1, 2, 3])"], ["import pandas as pd\ndf = pd.DataFrame([1, 2, 3]", "import pandas as pd\ndf = pd.Data([1, 2, 3])"]),

    # Ошибки в работе с JSON
    Issue("Ошибка в преобразовании в JSON", "import json\njson.dumps({'key': 'value'})", ["import json\njson.dumps({'key': 'value'})"], ["import json\njson.dumps({'key': 'value'}", "import json\njson.dump({'key': 'value'})"]),
    Issue("Ошибка в чтении JSON", "import json\njson.loads('{\"key\": \"value\"}')", ["import json\njson.loads('{\"key\": \"value\"}')"], ["import json\njson.loads('{\"key\": \"value\"}", "import json\njson.load('{\"key\": \"value\"}')"]),

    # Ошибки в работе с API
    Issue("Ошибка в запросе к API", "import requests\nrequests.get('https://api.example.com')", ["import requests\nrequests.get('https://api.example.com')"], ["import requests\nrequests.get('https://api.example.com'", "import requests\nrequests.get('https://api.example.com')"]),
    Issue("Ошибка в обработке ответа API", "import requests\nresponse = requests.get('https://api.example.com')\nprint(response.json())", ["import requests\nresponse = requests.get('https://api.example.com')\nprint(response.json())"], ["import requests\nresponse = requests.get('https://api.example.com')\nprint(response.json)", "import requests\nresponse = requests.get('https://api.example.com')\nprint(response.json())"]),

    # Ошибки в работе с базами данных
    Issue("Ошибка в подключении к базе данных", "import sqlite3\nconn = sqlite3.connect('example.db')", ["import sqlite3\nconn = sqlite3.connect('example.db')"], ["import sqlite3\nconn = sqlite3.connect('example.db'", "import sqlite3\nconn = sqlite3.connect('example.db')"]),
    Issue("Ошибка в выполнении SQL-запроса", "import sqlite3\nconn = sqlite3.connect('example.db')\ncursor = conn.cursor()\ncursor.execute('SELECT * FROM users')", ["import sqlite3\nconn = sqlite3.connect('example.db')\ncursor = conn.cursor()\ncursor.execute('SELECT * FROM users')"], ["import sqlite3\nconn = sqlite3.connect('example.db')\ncursor = conn.cursor()\ncursor.execute('SELECT * FROM users'", "import sqlite3\nconn = sqlite3.connect('example.db')\ncursor = conn.cursor()\ncursor.execute('SELECT * FROM users')"]),

    # Ошибки в работе с асинхронным кодом
    Issue("Ошибка в определении асинхронной функции", "async def my_func():\n    await asyncio.sleep(1)", ["async def my_func():\n    await asyncio.sleep(1)"], ["async def my_func():\n    await asyncio.sleep(1)", "async def my_func()\n    await asyncio.sleep(1)"]),
    Issue("Ошибка в вызове асинхронной функции", "import asyncio\nasyncio.run(my_func())", ["import asyncio\nasyncio.run(my_func())"], ["import asyncio\nasyncio.run(my_func()", "import asyncio\nasyncio.run(my_func())"]),

    # Ошибки в работе с декораторами
    Issue("Ошибка в определении декоратора", "def my_decorator(func):\n    def wrapper(*args, **kwargs):\n        return func(*args, **kwargs)\n    return wrapper", ["def my_decorator(func):\n    def wrapper(*args, **kwargs):\n        return func(*args, **kwargs)\n    return wrapper"], ["def my_decorator(func):\n    def wrapper(*args, **kwargs):\n        return func(*args, **kwargs)\n    return wrapper", "def my_decorator(func):\n    def wrapper(*args, **kwargs):\n        return func(*args, **kwargs)\n    return wrapper"]),
    Issue("Ошибка в использовании декоратора", "@my_decorator\ndef my_func():\n    pass", ["@my_decorator\ndef my_func():\n    pass"], ["@my_decorator\ndef my_func():\n    pass", "@my_decorator\ndef my_func()\n    pass"]),

    # Ошибки в работе с генераторами
    Issue("Ошибка в определении генератора", "def my_gen():\n    yield 1\n    yield 2", ["def my_gen():\n    yield 1\n    yield 2"], ["def my_gen():\n    yield 1\n    yield 2", "def my_gen()\n    yield 1\n    yield 2"]),
    Issue("Ошибка в использовании генератора", "for i in my_gen():\n    print(i)", ["for i in my_gen():\n    print(i)"], ["for i in my_gen():\n    print(i)", "for i in my_gen()\n    print(i)"]),

    # Ошибки в работе с контекстными менеджерами
    Issue("Ошибка в использовании контекстного менеджера", "with open('test.txt', 'r') as file:\n    print(file.read())", ["with open('test.txt', 'r') as file:\n    print(file.read())"], ["with open('test.txt', 'r') as file:\n    print(file.read())", "with open('test.txt', 'r') as file\n    print(file.read())"]),
    Issue("Ошибка в определении контекстного менеджера", "class MyContextManager:\n    def __enter__(self):\n        return self\n    def __exit__(self, exc_type, exc_val, exc_tb):\n        pass", ["class MyContextManager:\n    def __enter__(self):\n        return self\n    def __exit__(self, exc_type, exc_val, exc_tb):\n        pass"], ["class MyContextManager:\n    def __enter__(self):\n        return self\n    def __exit__(self, exc_type, exc_val, exc_tb):\n        pass", "class MyContextManager:\n    def __enter__(self):\n        return self\n    def __exit__(self, exc_type, exc_val, exc_tb)\n        pass"]),

    # Ошибки в работе с множествами
    Issue("Ошибка в определении множества", "my_set = {1, 2, 3}", ["my_set = {1, 2, 3}"], ["my_set = {1, 2, 3", "my_set = [1, 2, 3]"]),
    Issue("Ошибка в работе с множествами", "my_set.add(4)", ["my_set.add(4)"], ["my_set.add(4", "my_set.append(4)"]),

    # Ошибки в работе с кортежами
    Issue("Ошибка в определении кортежа", "my_tuple = (1, 2, 3)", ["my_tuple = (1, 2, 3)"], ["my_tuple = (1, 2, 3", "my_tuple = [1, 2, 3]"]),
    Issue("Ошибка в работе с кортежами", "print(my_tuple[0])", ["print(my_tuple[0])"], ["print(my_tuple[0]", "print(my_tuple(0))"]),

    # Ошибки в работе с классами данных
    Issue("Ошибка в определении класса данных", "from dataclasses import dataclass\n@dataclass\nclass MyClass:\n    x: int\n    y: int", ["from dataclasses import dataclass\n@dataclass\nclass MyClass:\n    x: int\n    y: int"], ["from dataclasses import dataclass\n@dataclass\nclass MyClass:\n    x: int\n    y: int", "from dataclasses import dataclass\n@dataclass\nclass MyClass:\n    x: int\n    y: int"]),
    Issue("Ошибка в использовании класса данных", "obj = MyClass(1, 2)", ["obj = MyClass(1, 2)"], ["obj = MyClass(1, 2", "obj = MyClass(1, 2)"]),

    # Ошибки в работе с модулем datetime
    Issue("Ошибка в использовании модуля datetime", "from datetime import datetime\nprint(datetime.now())", ["from datetime import datetime\nprint(datetime.now())"], ["from datetime import datetime\nprint(datetime.now()", "from datetime import datetime\nprint(datetime.now())"]),
    Issue("Ошибка в форматировании даты", "from datetime import datetime\nprint(datetime.now().strftime('%Y-%m-%d'))", ["from datetime import datetime\nprint(datetime.now().strftime('%Y-%m-%d'))"], ["from datetime import datetime\nprint(datetime.now().strftime('%Y-%m-%d'", "from datetime import datetime\nprint(datetime.now().strftime('%Y-%m-%d'))"]),

    # Ошибки в работе с модулем os
    Issue("Ошибка в использовании модуля os", "import os\nprint(os.getcwd())", ["import os\nprint(os.getcwd())"], ["import os\nprint(os.getcwd()", "import os\nprint(os.getcwd())"]),
    Issue("Ошибка в работе с путями", "import os\nprint(os.path.join('dir', 'file.txt'))", ["import os\nprint(os.path.join('dir', 'file.txt'))"], ["import os\nprint(os.path.join('dir', 'file.txt'", "import os\nprint(os.path.join('dir', 'file.txt'))"]),

    # Ошибки в работе с модулем sys
    Issue("Ошибка в использовании модуля sys", "import sys\nprint(sys.version)", ["import sys\nprint(sys.version)"], ["import sys\nprint(sys.version", "import sys\nprint(sys.version))"]),
    Issue("Ошибка в работе с аргументами командной строки", "import sys\nprint(sys.argv[0])", ["import sys\nprint(sys.argv[0])"], ["import sys\nprint(sys.argv[0]", "import sys\nprint(sys.argv[0))"]),

    # Ошибки в работе с модулем re
    Issue("Ошибка в использовании модуля re", "import re\nprint(re.match('^a', 'abc'))", ["import re\nprint(re.match('^a', 'abc'))"], ["import re\nprint(re.match('^a', 'abc'", "import re\nprint(re.match('^a', 'abc'))"]),
    Issue("Ошибка в работе с регулярными выражениями", "import re\nprint(re.sub('a', 'b', 'abc'))", ["import re\nprint(re.sub('a', 'b', 'abc'))"], ["import re\nprint(re.sub('a', 'b', 'abc'", "import re\nprint(re.sub('a', 'b', 'abc'))"]),

    # Ошибки в работе с модулем itertools
    Issue("Ошибка в использовании модуля itertools", "import itertools\nprint(list(itertools.islice(range(10), 5)))", ["import itertools\nprint(list(itertools.islice(range(10), 5)))"], ["import itertools\nprint(list(itertools.islice(range(10), 5)", "import itertools\nprint(list(itertools.islice(range(10), 5))"]),
    Issue("Ошибка в работе с itertools", "import itertools\nprint(list(itertools.chain([1, 2], [3, 4])))", ["import itertools\nprint(list(itertools.chain([1, 2], [3, 4])))"], ["import itertools\nprint(list(itertools.chain([1, 2], [3, 4]", "import itertools\nprint(list(itertools.chain([1, 2], [3, 4])))"]),

    # Ошибки в работе с модулем functools
    Issue("Ошибка в использовании модуля functools", "from functools import reduce\nprint(reduce(lambda x, y: x + y, [1, 2, 3]))", ["from functools import reduce\nprint(reduce(lambda x, y: x + y, [1, 2, 3]))"], ["from functools import reduce\nprint(reduce(lambda x, y: x + y, [1, 2, 3]", "from functools import reduce\nprint(reduce(lambda x, y: x + y, [1, 2, 3]))"]),
    Issue("Ошибка в работе с functools", "from functools import lru_cache\n@lru_cache\ndef my_func(x):\n    return x * 2", ["from functools import lru_cache\n@lru_cache\ndef my_func(x):\n    return x * 2"], ["from functools import lru_cache\n@lru_cache\ndef my_func(x):\n    return x * 2", "from functools import lru_cache\n@lru_cache\ndef my_func(x)\n    return x * 2"]),

    # Ошибки в работе с модулем collections
    Issue("Ошибка в использовании модуля collections", "from collections import defaultdict\nmy_dict = defaultdict(int)\nmy_dict['key'] += 1", ["from collections import defaultdict\nmy_dict = defaultdict(int)\nmy_dict['key'] += 1"], ["from collections import defaultdict\nmy_dict = defaultdict(int)\nmy_dict['key'] += 1", "from collections import defaultdict\nmy_dict = defaultdict(int)\nmy_dict['key'] += 1"]),
    Issue("Ошибка в работе с collections", "from collections import Counter\nprint(Counter('abc'))", ["from collections import Counter\nprint(Counter('abc'))"], ["from collections import Counter\nprint(Counter('abc'", "from collections import Counter\nprint(Counter('abc'))"]),

    # Ошибки в работе с модулем threading
    Issue("Ошибка в использовании модуля threading", "import threading\nthread = threading.Thread(target=my_func)\nthread.start()", ["import threading\nthread = threading.Thread(target=my_func)\nthread.start()"], ["import threading\nthread = threading.Thread(target=my_func)\nthread.start()", "import threading\nthread = threading.Thread(target=my_func)\nthread.start()"]),
    Issue("Ошибка в работе с threading", "import threading\nlock = threading.Lock()\nlock.acquire()\nlock.release()", ["import threading\nlock = threading.Lock()\nlock.acquire()\nlock.release()"], ["import threading\nlock = threading.Lock()\nlock.acquire()\nlock.release()", "import threading\nlock = threading.Lock()\nlock.acquire()\nlock.release()"]),

    # Ошибки в работе с модулем multiprocessing
    Issue("Ошибка в использовании модуля multiprocessing", "from multiprocessing import Process\np = Process(target=my_func)\np.start()", ["from multiprocessing import Process\np = Process(target=my_func)\np.start()"], ["from multiprocessing import Process\np = Process(target=my_func)\np.start()", "from multiprocessing import Process\np = Process(target=my_func)\np.start()"]),
    Issue("Ошибка в работе с multiprocessing", "from multiprocessing import Pool\nwith Pool(5) as p:\n    p.map(my_func, [1, 2, 3])", ["from multiprocessing import Pool\nwith Pool(5) as p:\n    p.map(my_func, [1, 2, 3])"], ["from multiprocessing import Pool\nwith Pool(5) as p:\n    p.map(my_func, [1, 2, 3]", "from multiprocessing import Pool\nwith Pool(5) as p:\n    p.map(my_func, [1, 2, 3])"]),

    # Ошибки в работе с модулем asyncio
    Issue("Ошибка в использовании модуля asyncio", "import asyncio\nasync def my_func():\n    await asyncio.sleep(1)\nasyncio.run(my_func())", ["import asyncio\nasync def my_func():\n    await asyncio.sleep(1)\nasyncio.run(my_func())"], ["import asyncio\nasync def my_func():\n    await asyncio.sleep(1)\nasyncio.run(my_func()", "import asyncio\nasync def my_func():\n    await asyncio.sleep(1)\nasyncio.run(my_func())"]),
    Issue("Ошибка в работе с asyncio", "import asyncio\nasync def my_func():\n    await asyncio.sleep(1)\nasyncio.run(my_func())", ["import asyncio\nasync def my_func():\n    await asyncio.sleep(1)\nasyncio.run(my_func())"], ["import asyncio\nasync def my_func():\n    await asyncio.sleep(1)\nasyncio.run(my_func()", "import asyncio\nasync def my_func():\n    await asyncio.sleep(1)\nasyncio.run(my_func())"]),

    # Ошибки в работе с модулем logging
    Issue("Ошибка в использовании модуля logging", "import logging\nlogging.basicConfig(level=logging.INFO)\nlogging.info('Hello')", ["import logging\nlogging.basicConfig(level=logging.INFO)\nlogging.info('Hello')"], ["import logging\nlogging.basicConfig(level=logging.INFO)\nlogging.info('Hello'", "import logging\nlogging.basicConfig(level=logging.INFO)\nlogging.info('Hello')"]),
    Issue("Ошибка в работе с logging", "import logging\nlogging.basicConfig(level=logging.INFO)\nlogging.info('Hello')", ["import logging\nlogging.basicConfig(level=logging.INFO)\nlogging.info('Hello')"], ["import logging\nlogging.basicConfig(level=logging.INFO)\nlogging.info('Hello'", "import logging\nlogging.basicConfig(level=logging.INFO)\nlogging.info('Hello')"]),

    # Ошибки в работе с модулем unittest
    Issue("Ошибка в использовании модуля unittest", "import unittest\nclass MyTest(unittest.TestCase):\n    def test_example(self):\n        self.assertEqual(1, 1)", ["import unittest\nclass MyTest(unittest.TestCase):\n    def test_example(self):\n        self.assertEqual(1, 1)"], ["import unittest\nclass MyTest(unittest.TestCase):\n    def test_example(self):\n        self.assertEqual(1, 1)", "import unittest\nclass MyTest(unittest.TestCase):\n    def test_example(self)\n        self.assertEqual(1, 1)"]),
    Issue("Ошибка в работе с unittest", "import unittest\nclass MyTest(unittest.TestCase):\n    def test_example(self):\n        self.assertEqual(1, 1)", ["import unittest\nclass MyTest(unittest.TestCase):\n    def test_example(self):\n        self.assertEqual(1, 1)"], ["import unittest\nclass MyTest(unittest.TestCase):\n    def test_example(self):\n        self.assertEqual(1, 1)", "import unittest\nclass MyTest(unittest.TestCase):\n    def test_example(self)\n        self.assertEqual(1, 1)"]),

    # Ошибки в работе с модулем pytest
    Issue("Ошибка в использовании модуля pytest", "import pytest\n@pytest.fixture\ndef my_fixture():\n    return 1", ["import pytest\n@pytest.fixture\ndef my_fixture():\n    return 1"], ["import pytest\n@pytest.fixture\ndef my_fixture():\n    return 1", "import pytest\n@pytest.fixture\ndef my_fixture()\n    return 1"]),
    Issue("Ошибка в работе с pytest", "import pytest\n@pytest.fixture\ndef my_fixture():\n    return 1", ["import pytest\n@pytest.fixture\ndef my_fixture():\n    return 1"], ["import pytest\n@pytest.fixture\ndef my_fixture():\n    return 1", "import pytest\n@pytest.fixture\ndef my_fixture()\n    return 1"]),

    # Ошибки в работе с модулем argparse
    Issue("Ошибка в использовании модуля argparse", "import argparse\nparser = argparse.ArgumentParser()\nparser.add_argument('--name', type=str)\nargs = parser.parse_args()", ["import argparse\nparser = argparse.ArgumentParser()\nparser.add_argument('--name', type=str)\nargs = parser.parse_args()"], ["import argparse\nparser = argparse.ArgumentParser()\nparser.add_argument('--name', type=str)\nargs = parser.parse_args()", "import argparse\nparser = argparse.ArgumentParser()\nparser.add_argument('--name', type=str)\nargs = parser.parse_args()"]),
    Issue("Ошибка в работе с argparse", "import argparse\nparser = argparse.ArgumentParser()\nparser.add_argument('--name', type=str)\nargs = parser.parse_args()", ["import argparse\nparser = argparse.ArgumentParser()\nparser.add_argument('--name', type=str)\nargs = parser.parse_args()"], ["import argparse\nparser = argparse.ArgumentParser()\nparser.add_argument('--name', type=str)\nargs = parser.parse_args()", "import argparse\nparser = argparse.ArgumentParser()\nparser.add_argument('--name', type=str)\nargs = parser.parse_args()"]),

    # Ошибки в работе с модулем subprocess
    Issue("Ошибка в использовании модуля subprocess", "import subprocess\nsubprocess.run(['ls', '-l'])", ["import subprocess\nsubprocess.run(['ls', '-l'])"], ["import subprocess\nsubprocess.run(['ls', '-l']", "import subprocess\nsubprocess.run(['ls', '-l'])"]),
    Issue("Ошибка в работе с subprocess", "import subprocess\nsubprocess.run(['ls', '-l'])", ["import subprocess\nsubprocess.run(['ls', '-l'])"], ["import subprocess\nsubprocess.run(['ls', '-l']", "import subprocess\nsubprocess.run(['ls', '-l'])"]),

    # Ошибки в работе с модулем shutil
    Issue("Ошибка в использовании модуля shutil", "import shutil\nshutil.copy('src.txt', 'dst.txt')", ["import shutil\nshutil.copy('src.txt', 'dst.txt')"], ["import shutil\nshutil.copy('src.txt', 'dst.txt'", "import shutil\nshutil.copy('src.txt', 'dst.txt')"]),
    Issue("Ошибка в работе с shutil", "import shutil\nshutil.copy('src.txt', 'dst.txt')", ["import shutil\nshutil.copy('src.txt', 'dst.txt')"], ["import shutil\nshutil.copy('src.txt', 'dst.txt'", "import shutil\nshutil.copy('src.txt', 'dst.txt')"]),

    # Ошибки в работе с модулем pathlib
    Issue("Ошибка в использовании модуля pathlib", "from pathlib import Path\np = Path('file.txt')\nprint(p.read_text())", ["from pathlib import Path\np = Path('file.txt')\nprint(p.read_text())"], ["from pathlib import Path\np = Path('file.txt')\nprint(p.read_text()", "from pathlib import Path\np = Path('file.txt')\nprint(p.read_text())"]),
    Issue("Ошибка в работе с pathlib", "from pathlib import Path\np = Path('file.txt')\nprint(p.read_text())", ["from pathlib import Path\np = Path('file.txt')\nprint(p.read_text())"], ["from pathlib import Path\np = Path('file.txt')\nprint(p.read_text()", "from pathlib import Path\np = Path('file.txt')\nprint(p.read_text())"]),

    # Ошибки в работе с модулем tempfile
    Issue("Ошибка в использовании модуля tempfile", "import tempfile\nwith tempfile.NamedTemporaryFile() as f:\n    f.write(b'data')", ["import tempfile\nwith tempfile.NamedTemporaryFile() as f:\n    f.write(b'data')"], ["import tempfile\nwith tempfile.NamedTemporaryFile() as f:\n    f.write(b'data'", "import tempfile\nwith tempfile.NamedTemporaryFile() as f:\n    f.write(b'data')"]),
    Issue("Ошибка в работе с tempfile", "import tempfile\nwith tempfile.NamedTemporaryFile() as f:\n    f.write(b'data')", ["import tempfile\nwith tempfile.NamedTemporaryFile() as f:\n    f.write(b'data')"], ["import tempfile\nwith tempfile.NamedTemporaryFile() as f:\n    f.write(b'data'", "import tempfile\nwith tempfile.NamedTemporaryFile() as f:\n    f.write(b'data')"]),

    # Ошибки в работе с модулем glob
    Issue("Ошибка в использовании модуля glob", "import glob\nprint(glob.glob('*.txt'))", ["import glob\nprint(glob.glob('*.txt'))"], ["import glob\nprint(glob.glob('*.txt'", "import glob\nprint(glob.glob('*.txt'))"]),
    Issue("Ошибка в работе с glob", "import glob\nprint(glob.glob('*.txt'))", ["import glob\nprint(glob.glob('*.txt'))"], ["import glob\nprint(glob.glob('*.txt'", "import glob\nprint(glob.glob('*.txt'))"]),

    # Ошибки в работе с модулем fnmatch
    Issue("Ошибка в использовании модуля fnmatch", "import fnmatch\nprint(fnmatch.fnmatch('file.txt', '*.txt'))", ["import fnmatch\nprint(fnmatch.fnmatch('file.txt', '*.txt'))"], ["import fnmatch\nprint(fnmatch.fnmatch('file.txt', '*.txt'", "import fnmatch\nprint(fnmatch.fnmatch('file.txt', '*.txt'))"]),
    Issue("Ошибка в работе с fnmatch", "import fnmatch\nprint(fnmatch.fnmatch('file.txt', '*.txt'))", ["import fnmatch\nprint(fnmatch.fnmatch('file.txt', '*.txt'))"], ["import fnmatch\nprint(fnmatch.fnmatch('file.txt', '*.txt'", "import fnmatch\nprint(fnmatch.fnmatch('file.txt', '*.txt'))"]),

    # Ошибки в работе с модулем zipfile
    Issue("Ошибка в использовании модуля zipfile", "import zipfile\nwith zipfile.ZipFile('archive.zip', 'r') as z:\n    z.extractall()", ["import zipfile\nwith zipfile.ZipFile('archive.zip', 'r') as z:\n    z.extractall()"], ["import zipfile\nwith zipfile.ZipFile('archive.zip', 'r') as z:\n    z.extractall()", "import zipfile\nwith zipfile.ZipFile('archive.zip', 'r') as z\n    z.extractall()"]),
    Issue("Ошибка в работе с zipfile", "import zipfile\nwith zipfile.ZipFile('archive.zip', 'r') as z:\n    z.extractall()", ["import zipfile\nwith zipfile.ZipFile('archive.zip', 'r') as z:\n    z.extractall()"], ["import zipfile\nwith zipfile.ZipFile('archive.zip', 'r') as z:\n    z.extractall()", "import zipfile\nwith zipfile.ZipFile('archive.zip', 'r') as z\n    z.extractall()"]),

    # Ошибки в работе с модулем tarfile
    Issue("Ошибка в использовании модуля tarfile", "import tarfile\nwith tarfile.open('archive.tar.gz', 'r:gz') as t:\n    t.extractall()", ["import tarfile\nwith tarfile.open('archive.tar.gz', 'r:gz') as t:\n    t.extractall()"], ["import tarfile\nwith tarfile.open('archive.tar.gz', 'r:gz') as t:\n    t.extractall()", "import tarfile\nwith tarfile.open('archive.tar.gz', 'r:gz') as t\n    t.extractall()"]),
    Issue("Ошибка в работе с tarfile", "import tarfile\nwith tarfile.open('archive.tar.gz', 'r:gz') as t:\n    t.extractall()", ["import tarfile\nwith tarfile.open('archive.tar.gz', 'r:gz') as t:\n    t.extractall()"], ["import tarfile\nwith tarfile.open('archive.tar.gz', 'r:gz') as t:\n    t.extractall()", "import tarfile\nwith tarfile.open('archive.tar.gz', 'r:gz') as t\n    t.extractall()"]),

    # Ошибки в работе с модулем csv
    Issue("Ошибка в использовании модуля csv", "import csv\nwith open('data.csv', 'r') as f:\n    reader = csv.reader(f)\n    for row in reader:\n        print(row)", ["import csv\nwith open('data.csv', 'r') as f:\n    reader = csv.reader(f)\n    for row in reader:\n        print(row)"], ["import csv\nwith open('data.csv', 'r') as f:\n    reader = csv.reader(f)\n    for row in reader:\n        print(row)", "import csv\nwith open('data.csv', 'r') as f\n    reader = csv.reader(f)\n    for row in reader:\n        print(row)"]),
    Issue("Ошибка в работе с csv", "import csv\nwith open('data.csv', 'r') as f:\n    reader = csv.reader(f)\n    for row in reader:\n        print(row)", ["import csv\nwith open('data.csv', 'r') as f:\n    reader = csv.reader(f)\n    for row in reader:\n        print(row)"], ["import csv\nwith open('data.csv', 'r') as f:\n    reader = csv.reader(f)\n    for row in reader:\n        print(row)", "import csv\nwith open('data.csv', 'r') as f\n    reader = csv.reader(f)\n    for row in reader:\n        print(row)"]),

    # Ошибки в работе с модулем json
    Issue("Ошибка в использовании модуля json", "import json\nprint(json.loads('{\"key\": \"value\"}'))", ["import json\nprint(json.loads('{\"key\": \"value\"}'))"], ["import json\nprint(json.loads('{\"key\": \"value\"}'", "import json\nprint(json.loads('{\"key\": \"value\"}'))"]),
    Issue("Ошибка в работе с json", "import json\nprint(json.loads('{\"key\": \"value\"}'))", ["import json\nprint(json.loads('{\"key\": \"value\"}'))"], ["import json\nprint(json.loads('{\"key\": \"value\"}'", "import json\nprint(json.loads('{\"key\": \"value\"}'))"]),

    # Ошибки в работе с модулем pickle
    Issue("Ошибка в использовании модуля pickle", "import pickle\nwith open('data.pkl', 'rb') as f:\n    data = pickle.load(f)", ["import pickle\nwith open('data.pkl', 'rb') as f:\n    data = pickle.load(f)"], ["import pickle\nwith open('data.pkl', 'rb') as f:\n    data = pickle.load(f)", "import pickle\nwith open('data.pkl', 'rb') as f\n    data = pickle.load(f)"]),
    Issue("Ошибка в работе с pickle", "import pickle\nwith open('data.pkl', 'rb') as f:\n    data = pickle.load(f)", ["import pickle\nwith open('data.pkl', 'rb') as f:\n    data = pickle.load(f)"], ["import pickle\nwith open('data.pkl', 'rb') as f:\n    data = pickle.load(f)", "import pickle\nwith open('data.pkl', 'rb') as f\n    data = pickle.load(f)"]),

    # Ошибки в работе с модулем shelve
    Issue("Ошибка в использовании модуля shelve", "import shelve\nwith shelve.open('data') as db:\n    db['key'] = 'value'", ["import shelve\nwith shelve.open('data') as db:\n    db['key'] = 'value'"], ["import shelve\nwith shelve.open('data') as db:\n    db['key'] = 'value'", "import shelve\nwith shelve.open('data') as db\n    db['key'] = 'value'"]),
    Issue("Ошибка в работе с shelve", "import shelve\nwith shelve.open('data') as db:\n    db['key'] = 'value'", ["import shelve\nwith shelve.open('data') as db:\n    db['key'] = 'value'"], ["import shelve\nwith shelve.open('data') as db:\n    db['key'] = 'value'", "import shelve\nwith shelve.open('data') as db\n    db['key'] = 'value'"]),

    # Ошибки в работе с модулем configparser
    Issue("Ошибка в использовании модуля configparser", "import configparser\nconfig = configparser.ConfigParser()\nconfig.read('config.ini')", ["import configparser\nconfig = configparser.ConfigParser()\nconfig.read('config.ini')"], ["import configparser\nconfig = configparser.ConfigParser()\nconfig.read('config.ini'", "import configparser\nconfig = configparser.ConfigParser()\nconfig.read('config.ini')"]),
    Issue("Ошибка в работе с configparser", "import configparser\nconfig = configparser.ConfigParser()\nconfig.read('config.ini')", ["import configparser\nconfig = configparser.ConfigParser()\nconfig.read('config.ini')"], ["import configparser\nconfig = configparser.ConfigParser()\nconfig.read('config.ini'", "import configparser\nconfig = configparser.ConfigParser()\nconfig.read('config.ini')"]),

    # Ошибки в работе с модулем argparse
    Issue("Ошибка в использовании модуля argparse", "import argparse\nparser = argparse.ArgumentParser()\nparser.add_argument('--name', type=str)\nargs = parser.parse_args()", ["import argparse\nparser = argparse.ArgumentParser()\nparser.add_argument('--name', type=str)\nargs = parser.parse_args()"], ["import argparse\nparser = argparse.ArgumentParser()\nparser.add_argument('--name', type=str)\nargs = parser.parse_args()", "import argparse\nparser = argparse.ArgumentParser()\nparser.add_argument('--name', type=str)\nargs = parser.parse_args()"]),
    Issue("Ошибка в работе с argparse", "import argparse\nparser = argparse.ArgumentParser()\nparser.add_argument('--name', type=str)\nargs = parser.parse_args()", ["import argparse\nparser = argparse.ArgumentParser()\nparser.add_argument('--name', type=str)\nargs = parser.parse_args()"], ["import argparse\nparser = argparse.ArgumentParser()\nparser.add_argument('--name', type=str)\nargs = parser.parse_args()", "import argparse\nparser = argparse.ArgumentParser()\nparser.add_argument('--name', type=str)\nargs = parser.parse_args()"]),

    # Ошибки в работе с модулем logging
    Issue("Ошибка в использовании модуля logging", "import logging\nlogging.basicConfig(level=logging.INFO)\nlogging.info('Hello')", ["import logging\nlogging.basicConfig(level=logging.INFO)\nlogging.info('Hello')"], ["import logging\nlogging.basicConfig(level=logging.INFO)\nlogging.info('Hello'", "import logging\nlogging.basicConfig(level=logging.INFO)\nlogging.info('Hello')"]),
    Issue("Ошибка в работе с logging", "import logging\nlogging.basicConfig(level=logging.INFO)\nlogging.info('Hello')", ["import logging\nlogging.basicConfig(level=logging.INFO)\nlogging.info('Hello')"], ["import logging\nlogging.basicConfig(level=logging.INFO)\nlogging.info('Hello'", "import logging\nlogging.basicConfig(level=logging.INFO)\nlogging.info('Hello')"]),

    # Ошибки в работе с модулем unittest
    Issue("Ошибка в использовании модуля unittest", "import unittest\nclass MyTest(unittest.TestCase):\n    def test_example(self):\n        self.assertEqual(1, 1)", ["import unittest\nclass MyTest(unittest.TestCase):\n    def test_example(self):\n        self.assertEqual(1, 1)"], ["import unittest\nclass MyTest(unittest.TestCase):\n    def test_example(self):\n        self.assertEqual(1, 1)", "import unittest\nclass MyTest(unittest.TestCase):\n    def test_example(self)\n        self.assertEqual(1, 1)"]),
    Issue("Ошибка в работе с unittest", "import unittest\nclass MyTest(unittest.TestCase):\n    def test_example(self):\n        self.assertEqual(1, 1)", ["import unittest\nclass MyTest(unittest.TestCase):\n    def test_example(self):\n        self.assertEqual(1, 1)"], ["import unittest\nclass MyTest(unittest.TestCase):\n    def test_example(self):\n        self.assertEqual(1, 1)", "import unittest\nclass MyTest(unittest.TestCase):\n    def test_example(self)\n        self.assertEqual(1, 1)"]),

    # Ошибки в работе с модулем pytest
    Issue("Ошибка в использовании модуля pytest", "import pytest\n@pytest.fixture\ndef my_fixture():\n    return 1", ["import pytest\n@pytest.fixture\ndef my_fixture():\n    return 1"], ["import pytest\n@pytest.fixture\ndef my_fixture():\n    return 1", "import pytest\n@pytest.fixture\ndef my_fixture()\n    return 1"]),
    Issue("Ошибка в работе с pytest", "import pytest\n@pytest.fixture\ndef my_fixture():\n    return 1", ["import pytest\n@pytest.fixture\ndef my_fixture():\n    return 1"], ["import pytest\n@pytest.fixture\ndef my_fixture():\n    return 1", "import pytest\n@pytest.fixture\ndef my_fixture()\n    return 1"]),

    # Ошибки в работе с модулем argparse
    Issue("Ошибка в использовании модуля argparse", "import argparse\nparser = argparse.ArgumentParser()\nparser.add_argument('--name', type=str)\nargs = parser.parse_args()", ["import argparse\nparser = argparse.ArgumentParser()\nparser.add_argument('--name', type=str)\nargs = parser.parse_args()"], ["import argparse\nparser = argparse.ArgumentParser()\nparser.add_argument('--name', type=str)\nargs = parser.parse_args()", "import argparse\nparser = argparse.ArgumentParser()\nparser.add_argument('--name', type=str)\nargs = parser.parse_args()"]),
    Issue("Ошибка в работе с argparse", "import argparse\nparser = argparse.ArgumentParser()\nparser.add_argument('--name', type=str)\nargs = parser.parse_args()", ["import argparse\nparser = argparse.ArgumentParser()\nparser.add_argument('--name', type=str)\nargs = parser.parse_args()"], ["import argparse\nparser = argparse.ArgumentParser()\nparser.add_argument('--name', type=str)\nargs = parser.parse_args()", "import argparse\nparser = argparse.ArgumentParser()\nparser.add_argument('--name', type=str)\nargs = parser.parse_args()"]),

    # Ошибки в работе с модулем subprocess
    Issue("Ошибка в использовании модуля subprocess", "import subprocess\nsubprocess.run(['ls', '-l'])", ["import subprocess\nsubprocess.run(['ls', '-l'])"], ["import subprocess\nsubprocess.run(['ls', '-l']", "import subprocess\nsubprocess.run(['ls', '-l'])"]),
    Issue("Ошибка в работе с subprocess", "import subprocess\nsubprocess.run(['ls', '-l'])", ["import subprocess\nsubprocess.run(['ls', '-l'])"], ["import subprocess\nsubprocess.run(['ls', '-l']", "import subprocess\nsubprocess.run(['ls', '-l'])"]),

    # Ошибки в работе с модулем shutil
    Issue("Ошибка в использовании модуля shutil", "import shutil\nshutil.copy('src.txt', 'dst.txt')", ["import shutil\nshutil.copy('src.txt', 'dst.txt')"], ["import shutil\nshutil.copy('src.txt', 'dst.txt'", "import shutil\nshutil.copy('src.txt', 'dst.txt')"]),
    Issue("Ошибка в работе с shutil", "import shutil\nshutil.copy('src.txt', 'dst.txt')", ["import shutil\nshutil.copy('src.txt', 'dst.txt')"], ["import shutil\nshutil.copy('src.txt', 'dst.txt'", "import shutil\nshutil.copy('src.txt', 'dst.txt')"]),

    # Ошибки в работе с модулем pathlib
    Issue("Ошибка в использовании модуля pathlib", "from pathlib import Path\np = Path('file.txt')\nprint(p.read_text())", ["from pathlib import Path\np = Path('file.txt')\nprint(p.read_text())"], ["from pathlib import Path\np = Path('file.txt')\nprint(p.read_text()", "from pathlib import Path\np = Path('file.txt')\nprint(p.read_text())"]),
    Issue("Ошибка в работе с pathlib", "from pathlib import Path\np = Path('file.txt')\nprint(p.read_text())", ["from pathlib import Path\np = Path('file.txt')\nprint(p.read_text())"], ["from pathlib import Path\np = Path('file.txt')\nprint(p.read_text()", "from pathlib import Path\np = Path('file.txt')\nprint(p.read_text())"]),

    # Ошибки в работе с модулем tempfile
    Issue("Ошибка в использовании модуля tempfile", "import tempfile\nwith tempfile.NamedTemporaryFile() as f:\n    f.write(b'data')", ["import tempfile\nwith tempfile.NamedTemporaryFile() as f:\n    f.write(b'data')"], ["import tempfile\nwith tempfile.NamedTemporaryFile() as f:\n    f.write(b'data'", "import tempfile\nwith tempfile.NamedTemporaryFile() as f:\n    f.write(b'data')"]),
    Issue("Ошибка в работе с tempfile", "import tempfile\nwith tempfile.NamedTemporaryFile() as f:\n    f.write(b'data')", ["import tempfile\nwith tempfile.NamedTemporaryFile() as f:\n    f.write(b'data')"], ["import tempfile\nwith tempfile.NamedTemporaryFile() as f:\n    f.write(b'data'", "import tempfile\nwith tempfile.NamedTemporaryFile() as f:\n    f.write(b'data')"]),

    # Ошибки в работе с модулем glob
    Issue("Ошибка в использовании модуля glob", "import glob\nprint(glob.glob('*.txt'))", ["import glob\nprint(glob.glob('*.txt'))"], ["import glob\nprint(glob.glob('*.txt'", "import glob\nprint(glob.glob('*.txt'))"]),
    Issue("Ошибка в работе с glob", "import glob\nprint(glob.glob('*.txt'))", ["import glob\nprint(glob.glob('*.txt'))"], ["import glob\nprint(glob.glob('*.txt'", "import glob\nprint(glob.glob('*.txt'))"]),
]

# Запуск игры
if __name__ == "__main__":
    root = tk.Tk()
    game = PythonJuniorGame(root)
    root.mainloop()
