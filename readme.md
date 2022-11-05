Данный проект является реализацией простого перцептрона с использованием ООП.
В качестве функции активации используем функцию Sigmoid.


В коде файла perceptron.py приведён пример переобучения перцептрона.
Попробуем с помощью перцентрона сделать модель логической функции XOR, если во входящей выборке будет 4 значения для обучения, то с увеличением кол-ва эпох мы получим ухудшение результата (функция example1). Однако, если мы уменьшим обучающую выборку до 3 значений, то с увеличением кол-ва эпох мы будем только улучшать наш результат (функция example2).

Более подробно эффект переобучения рассматривается в этой статье:
https://proproprogs.ru/neural_network/pereobuchenie-chto-eto-i-kak-etogo-izbezhat-kriterii-ostanova-obucheniya
