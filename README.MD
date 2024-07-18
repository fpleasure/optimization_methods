# О проекте

Реализация классических методов оптимизации на *Python*.

### Список доступных методов

- Дихотомия
- Метод золотого сечения
- Градиентный спуск

## Начало работы

### Используемые технологии

- [Python](https://www.python.org/)
- [Matplotlib](https://matplotlib.org/stable/)
- [Autograd](https://github.com/HIPS/autograd)
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

Для построения иллюстраций работы алгоритма используйте функцию *plot* из модуля *plots*:

```python
from optimizer import Optimizer
from src.plots.two_dimensional import plot

objective_function = lambda x: x[0] ** 2 + x[1] ** 2

opt = Optimizer("gradient_descent")
data = opt.optimize(objective_function, initial_guess=[3, 2], precsision=1e-3, callback=True)

figure = plot(data, objective_function)
figure.savefig("quadratic.png")
```

![Пример отрисовки](/img/quadratic.png)

### Тестирование

Для тестирования запустите следующий скрипт из директории *optimization_methods*:

```bash
python -m unittest discover -s src/tests -p "test_*.py"
```


## Контакты
- [Никита Королев](https://t.me/niki_korolev)
