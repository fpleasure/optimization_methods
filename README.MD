# О проекте

Реализация классических методов оптимизации на *Python*.

## Начало работы

### Используемые технологии

- [Python](https://www.python.org/)
- [Matplotlib](https://matplotlib.org/stable/)
- [NumPy](https://numpy.org/)

### Установка

Пошаговая инструкция для установки и запуска:

1. Клонируйте репозиторий:

    ```bash
    git clone https://github.com/fpleasure/optimization_methods.git
    ```

2. Перейдите в директорию проекта:

    ```bash
    cd optimization_methods
    ```

3. Создайте и активируйте виртуальное окружение:

    ```bash
    python -m venv venv
    source venv/bin/activate  # для Windows используйте `venv\Scripts\activate`
    ```

4. Установите необходимые зависимости:

    ```bash
    pip install -r requirements.txt
    ```

### Использование

Для минимизации функции используется класс *Optimizer*, в качестве обязательного аргумента конструктора передается название метода оптимизации. Нативно во всех методах пакета реализована минимизация функции. Пример работы с градиентным спуском:

```python
from optimizer import Optimizer

opt = Optimizer("gradient_descent")
minimum = opt.optimize(lambda x: x ** 2, initial_guess=2)
```

### Список доступных методов

- Градиентный спуск

## Контакты
- [Никита Королев](https://t.me/niki_korolev)
