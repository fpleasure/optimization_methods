# О проекте

Бибилотека методов оптимизации на *Python*. В ней реализованы как классические методы (GD, м. Ньютона, BFGS и т.д.), так и современные методы (Adagrad, RMSProp) [в разработке], а также доступны отрисовки работы алгоритмов и базовые тестовые функции (ф-ия Розенброка, Била, Матьяса).

### Используемые технологии

- [Python](https://www.python.org/)
- [Matplotlib](https://matplotlib.org/stable/)
- [Autograd](https://github.com/HIPS/autograd)
- [NumPy](https://numpy.org/)

### Список доступных методов

#### Одномерная минимизация
* Методы 0-го порядка
    + Дихотомия
    + Метод золотого сечения

#### Многомерная минимизация
* Методы 1-го порядка
    + Градиентный спуск с постоянным шагом
    + Наискорейший спуск
* Методы сопряженных направлений
    + Флетера-Ривса
    + Полака-Рибьера
* Методы 2-го порядка
    + Метод Ньютона
    + Усовершенствованный метод Ньютона
* Квазиньютоновские методы
    + BFGS
    + DFP
    + Метод Пауэлла
* Методы прямого поиска
    + Циклический покоординатный спуск
    + Хука-Дживса
    + Розенброка

#### Методы оптимизации в ML
[в разработке]

## Начало работы

### Содержание репозитория

```
├── examples                        # Примеры
├── img                             # Картинки для README 
├── tests                           # unit-тесты
├── optimization_methods
│   ├──optimizers                   # Оптимизаторы
│   │   ├──one_dimensional.py       # Одномерная минимизация
│   │   └──multidimensional.py      # Многомерная минимизация
│   ├──exceptions.py                # Исключения
│   ├──functions.py                 # Функции
│   ├──plotter.py                   # Отрисовка
│   └──result.py                    # Результаты работы оптимизатора
├── requirements.txt
└── README.md
```

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
    python3 -m venv venv
    source venv/bin/activate  # для Windows используйте `venv\Scripts\activate`
    ```

4. Установите необходимые зависимости:

    ```bash
    pip install -r requirements.txt
    ```

### Использование

#### Класс **OptimizationResult**
В данном классе хранятся результаты оптимизации: точка минимума, путь до минимума, статус работы алгоритма, количество итераций.

#### Классы минимизаторов
Все алгоритмы хранятся в папке *optimizers*, в *base.py* хранится абстрактный класс, в файле *one_dimensional.py* находятся алгоритмы одномерной минимизации, в файле *multidimensional.py* находятся алгоритмы многомерной оптимизации.

#### Класс **OptimizationPlotter**
Данный класс отрисовывает путь работы алгоритма, линии уровня, начальную и конечную точки.

### Примеры использования
Запускать код можно из корневой директории *optimization_methods*, желательно запускать его как модули. Тогда, например, чтобы запустить файл *examples/one_dimensional.py* пропишем из корневой директории

```bash
python3 -m examples.one_dimensional 
```

Рассмотрим сначала работу алгоритма одномерной минимизации. В качестве пути у результата одномерной минимизации хранится путь отрезков, которые получаются в ходе работы алгоритма.

```python
from optimization_methods.optimizers.one_dimensional import GoldenRatio


def f(x):
	return x ** 2

optimizer = GoldenRatio()
result = optimizer.minimize(f, *[-1, 1], tolerance=1e-1, max_iterations=300)
print(result)
```
Результат:
```bash
Optimization result:
solution: 0.021286236252208116
iterations: 7
trace: [array([-1,  1]), array([-0.23606798,  1.        ]), array([-0.23606798,  0.52786405]), array([-0.23606798,  0.23606798]), array([-0.05572809,  0.23606798]), array([-0.05572809,  0.1246118 ]), array([-0.05572809,  0.05572809]), array([-0.01315562,  0.05572809])]
status: Optimization successfully stoped.
```

Для многомерной оптимизации все аналогично, только вместо начального отрезка задается начальная точка. Все в той же папке примеров в файле *multidimensional.py*:

```python
import numpy as np
from optimization_methods.optimizers.multidimensional import Newton


def f(x):
	return 3 * x[0] ** 2 + 5 * x[1] ** 2

optimizer = Newton()
result = optimizer.minimize(f, np.array([-5.0, 2.0]), tolerance=1e-2)
print(result)

```
Результат:
```bash
Optimization result:
solution: [0. 0.]
iterations: 1
trace: [array([-5.,  2.]), array([0., 0.])]
status: Optimization successfully stoped.
```
Попробуем теперь отрисовать получившуюся траекторию: импортируем специальный класс.

```python
from optimization_methods.plotter import OptimizationPlotter
```

и отриусем:

```python
plot = OptimizationPlotter(result, f).plot()
plot.show()
```

![Пример отрисовки](/img/newton.png)

### Тестирование

Для тестирования запустите следующий скрипт из корневой директории:

```bash
python3 -m unittest discover tests
```


## Контакты
- [Никита Королев](https://t.me/niki_korolev)
