
import time
import textwrap
from typing import List
import copy
import joblib
import datetime
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
import pandas as pd
from networkx.drawing.nx_agraph import graphviz_layout


class fca_lattice:
    def __init__(self, df: pd.DataFrame):
        """
        Конструктор класса. Инициализирует основные свойства.
        :param df: Полный бинарный датафрейм, по которому будут определятся концепты.
        :param param: Целевой параметр из числа столбцов df. По умолчанию пустая строка.
        :param stack_intervals_count: Количество шагов для расчета концептов большого контекста. По умолчанию 100.
        Возможно по умолчанию лучше 0, чтобы иметь возможность простого рекурсивного расчета.
        TODO
        В идеале хотелось бы загружать исходную таблицу и накладывать фильтр по выбранному целевому параметру,
        для того чтобы вычислять концепты по сокращенной выборке, а оценки считать по полной.
        """
        self.context = df

        # определяем супремум и инфимум решетки
        self.concepts = [{'A': set(self.context.index), 'B': set()}, {'A': set(), 'B': set(self.context.columns)}]


        # проверить следующую строку
        self.threshold_base = len(self.context.index)

        # Множество концептов для быстрого расчета. Генерится только объем и хранится в виде кортежа (хешируемый тип)
        self.concepts_set = set()

        self.columns_len = len(self.context.columns)
        self.index_len = len(self.context.index)
        # Определяем переменные для интервального расчета концептов
        self.stack_intervals_count = 0
        self.stack_intervals = pd.DataFrame()
        self.stack = []

        # Предварительный расчет обемов для каждого столбца и содержаний для каждой строки. Ускоряет расчет концептов.
        self.context_derivation_0 = pd.Series(index=self.context.index, dtype='object')
        self.context_derivation_1 = pd.Series(index=self.context.columns, dtype='object')
        for i in range(0, self.index_len):
            self.context_derivation_0.iloc[i] = self.derivation(self.context.index[i], 0)
        for i in range(0, self.columns_len):
            self.context_derivation_1.iloc[i] = self.derivation(self.context.columns[i], 1)
        # Инициализация двунаправленного графа для представления решетки
        self.lattice = nx.DiGraph()

        # Инициализация двунаправленного графа для экспериментальной решетки с маркированными ребрами (пока не вышло) для ускорения
        # выполнения запросов к ИАМ. Надо бы разделить ИАМ от простого АФП и от Ассоциативных правил.
        # self.lbl_lattice = nx.DiGraph()

    def is_cannonical(self, column, new_a, r):

        for i in range(column, -1, -1):
            if self.context.columns[i] not in self.concepts[r]['B']:
                if new_a.issubset(self.context_derivation_1.iloc[i]):
                    return False
        return True

    def in_close(self, column: int, r: int, threshold=0.0):

        for j in range(column, self.columns_len):
            new_concept = {'A': self.context_derivation_1.iloc[j].intersection(self.concepts[r]['A']), 'B': set()}
            if len(new_concept['A']) == len(self.concepts[r]['A']):
                self.concepts[r]['B'].add(self.context.columns[j])
            else:
                if (len(new_concept['A']) != 0) and (len(new_concept['A']) > self.threshold_base * threshold):
                    if self.is_cannonical(j - 1, new_concept['A'], r):
                        new_concept['B'] = new_concept['B'].union(self.concepts[r]['B'])
                        new_concept['B'].add(self.context.columns[j])
                        self.concepts.append(new_concept)
                        self.in_close(j + 1, len(self.concepts) - 1, threshold)
        self.concepts_copy = copy.deepcopy(self.concepts)

    def __my_close__(self, column: int, concept_A: set, interval_number: int):
        """
        Оригинальный алгоритм поиска концептов в интервалах
        :param column: номер столбца
        :param concept_A: объем концепта как множество индексов строк
        :param interval_number: номер интервала расчета
        :return:
        """
        tp_concept_a = tuple(sorted(concept_A))
        if tp_concept_a not in self.concepts_set:
            self.concepts_set.add(tp_concept_a)

        for j in range(column, self.columns_len):
            new_concept_a = concept_A.intersection(self.context_derivation_1.iloc[j])
            new_concept_a_len = len(new_concept_a)
            tp_concept_a = tuple(sorted(new_concept_a))
            if (new_concept_a_len > self.stack_intervals.loc[interval_number, 'left']) & (
                    new_concept_a_len <= self.stack_intervals.loc[interval_number, 'right']):
                if tp_concept_a not in self.concepts_set:
                    self.concepts_set.add(tp_concept_a)
                    print('\r', len(self.concepts_set), end='')
                    self.__my_close__(j + 1, new_concept_a, interval_number)
            elif (new_concept_a_len <= self.stack_intervals.loc[interval_number, 'left']) & (new_concept_a_len > 0):
                # print('\r', new_concept_a_len, end='')
                ind = self.stack_intervals[(self.stack_intervals['left'] < new_concept_a_len) & (self.stack_intervals['right'] >= new_concept_a_len)].index.values[0]
                # добавление параметров в стек вызова
                if (tp_concept_a not in self.stack[ind]) or (self.stack[ind][tp_concept_a] > j+1):
                    self.stack[ind].update({tp_concept_a: j+1})

    def stack_my_close(self, step_count: int = 100):
        """
        Процедура интервального расчета концептов. Управление стеком параметров вызова функции __my_close__ по интервалам
        :param step_count: количество шагов расчета
        :return:
        """
        # Шаг расчета
        self.stack_intervals_count = step_count
        step = self.index_len / step_count
        # Интервалы для быстрого расчета концептов. Левая и правая границы.
        self.stack_intervals = self.stack_intervals.reindex(index=range(step_count))
        self.stack_intervals['left'] = [np.around(step * (step_count - i)) for i in range(1, step_count + 1)]
        self.stack_intervals['right'] = [np.around(step * (step_count - i)) for i in range(step_count)]
        # Стек параметров вызова функции для каждого интервала. Позваляет постепенно опускаться вглубь,
        # расчитывая сперва самые большие по объему концепты.
        self.stack = [{} for i in range(step_count)]

        concept_count = 0
        # добавление супремума как первого набора параметров вызова ункции расчета концептов в нулевом интервале
        self.stack[0].update({tuple(sorted(set(self.context.index))): 0})
        # проход по интервалам
        for i in range(step_count):
            # печать информации о списке параметров вызова в интервале
            print('\n', i,', interval: ', self.stack_intervals.loc[i, 'left'], ' - ', self.stack_intervals.loc[i, 'right'],
                  ', stack: ', len(self.stack[i]))
            # вызов функци расчета концептов с сохраненными параметрвами вызова
            for k in self.stack[i].keys():
                self.__my_close__(self.stack[i][k], set(k), i)
            # подсчет общего числа концептов
            concept_count = concept_count + len(self.concepts_set)
            print('concepts: ', len(self.concepts_set), '/', concept_count)
            # выгрузка найденных концептов в файл, очистка списка концептов и стека вызова для интервала
            joblib.dump(self.concepts_set, ".\\result\\concepts_set" + str(i) + ".joblib")
            self.concepts_set.clear()
            
    def read_concepts(self,num_concept_set:int):
        """
        Загрузка концептов расчитанных пошагово. Надо подумть как лучше сделать ,если количество шагов расчета
        не является свойстом решетки, а задается параметром
        :param num_concept_set:
        :return:
        """
        #выгрузка
        load_joblib = joblib.load(".\\result\\concepts_set" + str(num_concept_set) + ".joblib")
        #проверка на пустую выгрузку
        if load_joblib!=set():
            load_joblib = set(list(load_joblib)[0])
            B=set(self.context_derivation_0.index)
            B=load_joblib.intersection(B)
            B=self.context_derivation_0[list(B)].values
            B=list(B)
            final_B=B[0]
            for i in B:
                final_B=final_B.intersection(i)
            self.concepts = [{'A': set(load_joblib), 'B': final_B}]
        elements_index=list(self.concepts[0]['A'])
        elements_column=list(self.concepts[0]['B'])
        return self.context[elements_column].loc[elements_index]


    # def stack_concepts_repair(self, ):


    def derivation(self, q_val: str, axis):
        """
        Вычисляет по контексту множество штрих для одного элемента (строка или столбец)
        :param q_val: индекс столбца или строки
        :param axis: ось (1 - стобец, 0 - строка)
        :return: результат деривации (операции штрих)
        """
        if axis == 1:
            # поиск по измерениям (столбцам)
            tmp_df = self.context.loc[:, q_val]
        else:
            # поиск по показателям (строкам)
            tmp_df = self.context.loc[q_val, :]
        return set(tmp_df[tmp_df == 1].index)

    def multi_derivation(self, axis: int, combination_type: str, elements: List[str]):
        """
        Выполняет деривацию для нескольких элементов на основе указанного типа множества, оси, типа комбинации и элементов.
        :param set_type: Тип множества ('F' для формального или 'D' для производного)
        :param axis: Ось, по которой выполняется деривация (0 для строк, 1 для столбцов)
        :param combination_type: Тип комбинации ('AND' или 'OR')
        :param elements: Список элементов для деривации
        :return: Результат деривации
        """
        # Список для хранения полученных множеств
        derivations = []

        # Выполнение деривации для каждого элемента
        for element in elements:
            derived_set = self.derivation(element, axis)
            derivations.append(derived_set)

        # Комбинирование полученных множеств
        if combination_type == 'AND':
            combined_set = set.intersection(*derivations)
        elif combination_type == 'OR':
            combined_set = set.union(*derivations)
        else:
            print("Недопустимый тип комбинации. Допустимые значения: 'AND' или 'OR'.")
            return None

        # Возвращение объединенного множества
        return combined_set

    def multi_derivation_procedure(self, element: str, table: pd.DataFrame) -> pd.DataFrame:
        start_time = time.time()

        element = element.strip().lower()

        if element.startswith("f"):
            axis = 0
        elif element.startswith("d"):
            axis = 1
        else:
            print("⚠️ Неизвестный тип элемента:", element)
            return table  # возвращаем ту же таблицу без изменений

        elements = [element]

        result1 = self.multi_derivation(axis, "AND", elements)
        if result1 is None:
            return table

        result2 = self.multi_derivation(1 - axis, "OR", list(result1))
        if result2 is None:
            return table

        # фильтруем текущую таблицу
        if axis == 0:
            table = table.loc[:, table.columns.intersection(result1)]
        else:
            table = table.loc[table.index.intersection(result1), :]

        if (1 - axis) == 0:
            table = table.loc[:, table.columns.intersection(result2)]
        else:
            table = table.loc[table.index.intersection(result2), :]

        elapsed = time.time() - start_time


        return table, elapsed

    def print_indexes(self):
        # Вывод индексов столбцов
        print("Столбцы:", end=" ")
        print(", ".join(self.context.columns))

        # Вывод индексов строк
        print("Строки:", end=" ")
        print(", ".join(self.context.index))





    def find_reachable_concepts(lat):
        # Варианты выбора главной вершины:
        # supremum_node = 0  - супремум
        # infimum_node = 1   - инфимум
        print("Выберите главную вершину:")
        print("1. Супремум")
        print("2. Инфимум")

        starting_node_choice = input("Введите номер выбранной опции (1 или 2): ")
        if starting_node_choice == "1":
            starting_node = 0  # Супремум
            direction = "down"  # Направление обхода - вниз
        elif starting_node_choice == "2":
            starting_node = len(lat.concepts) - 1  # Инфимум
            direction = "up"  # Направление обхода - вверх

        num_edges = int(input("Введите количество ребер для поиска достижимых концептов: "))
        reachable_concepts = set()

        # Используем обход графа в глубину для нахождения всех достижимых концептов
        def dfs(node, depth, result, visited):
            # Если достигли нужной глубины или уже посещали этот узел, выходим из рекурсии
            if depth > num_edges or node in visited:
                return

            # Добавляем текущий узел в множество достижимых концептов
            result[depth].append(node)
            visited.add(node)
            reachable_concepts.add(node)

            # Определяем список соседей в зависимости от направления обхода
            neighbors = lat.lattice.successors(node) if direction == "down" else lat.lattice.predecessors(node)

            # Рекурсивно вызываем функцию dfs для соседей текущего узла
            for neighbor in neighbors:
                dfs(neighbor, depth + 1, result, visited)

        # Запускаем обход графа в глубину, начиная с указанной вершины
        result = {i: [] for i in range(num_edges + 1)}
        visited = set()
        dfs(starting_node, depth=0, result=result, visited=visited)

        print("Достижимые концепты:")
        for depth, concepts in result.items():
            print(f"Достижимые за {depth} ребер: {concepts}")
        for concept_idx in reachable_concepts:
            concept = lat.concepts[concept_idx]
            print(f"Концепт {concept_idx}: A = {concept['A']}, B = {concept['B']}")



    def process_reachable_concepts(lat):
        available_f_elements = set(lat.all_elements['F'])  # Все доступные элементы f
        available_d_elements = set(lat.all_elements['D'])  # Все доступные элементы d
        known_supremum = None
        known_infimum = None

        while True:
            element = input("Введите элемент (например, f2 или d1), или 'q' для выхода: ")

            if element.lower() == 'q':
                break

            element_type = element[0]  # Берем первую букву (f или d)

            if element_type.upper() == 'F':
                known_infimum = lat.find_specific_suitable_concept(lat, element, direction="up",known_infimum=known_infimum)
                if known_infimum is not None:
                    available_d_elements &= set(lat.concepts[known_infimum]['B'])  # Пересечение с доступными d
            elif element_type.upper() == 'D':
                known_supremum = lat.find_specific_suitable_concept(lat, element, direction="down",known_supremum=known_supremum)
                if known_supremum is not None:
                    available_f_elements &= set(lat.concepts[known_supremum]['A'])  # Пересечение с доступными f
            else:
                print("Недопустимый тип множества.")

        print("Завершение работы.")
    def find_specific_suitable_concept(lat, element, direction, known_supremum=None, known_infimum=None):

        if direction == "up":
            starting_node = len(lat.concepts) - 1  # Инфимум
            if known_infimum is not None:
                starting_node = known_infimum
        elif direction == "down":
            starting_node = 0  # Супремум
            if known_supremum is not None:
                starting_node = known_supremum
        else:
            print("Недопустимый тип множества.")
            return

        found_concept = None  # Флаг, который указывает, что нужный концепт найден

        def dfs(node, visited):
            nonlocal found_concept  # Объявляем, что будем использовать переменную из внешней области видимости

            # Если уже нашли нужный концепт, завершаем рекурсию
            if found_concept is not None:
                return

            visited.add(node)

            # Определяем список соседей в зависимости от направления обхода
            neighbors = lat.lattice.successors(node) if direction == "down" else lat.lattice.predecessors(node)

            # Проверяем, содержится ли элемент в текущем концепте
            concept = lat.concepts[node]
            if element in concept['A'] or element in concept['B']:
                found_concept = node  # Устанавливаем флаг, что нужный концепт найден
                return

            # Рекурсивно вызываем функцию dfs для соседей текущего узла
            for neighbor in neighbors:
                if neighbor not in visited:
                    dfs(neighbor, visited)

        dfs(starting_node, visited=set())


        return found_concept

    def fill_lattice(self):
        """
        Заполняет двунаправленный граф (решетку). Пересмотреть расчет ребер с инфимумом и генерацию лейблов ребер!!!
        :return:
        """
        # сортируем множество концептов по мощности объема. Навводила разные ключи в словарь, надо бы упорядочить.
        for i in range(len(self.concepts)):
            self.concepts[i]['W'] = len(self.concepts[i]['A'])
        self.concepts = sorted(self.concepts, key=lambda concept: concept['W'], reverse=True)

        for i in range(len(self.concepts)):
            self.lattice.add_node(i, ext_w=self.concepts[i]['W'],
                                  intent=','.join(str(s) for s in self.concepts[i]['B']))
            for j in range(i - 1, -1, -1):
                if (self.concepts[j]['B'].issubset(self.concepts[i]['B'])) & (
                        self.concepts[i]['A'].issubset(self.concepts[j]['A'])):
                    if not nx.has_path(self.lattice, j, i):
                        self.lattice.add_edge(j, i,
                                             add_d=','.join(str(s) for s in self.concepts[i]['B'] - self.concepts[j]['B']),
                                             add_m=','.join(str(s) for s in self.concepts[j]['A'] - self.concepts[i]['A']))

    def lat_draw(self):
        """
        Рисование решетки
        :return:
        """
        min_w = len(self.concepts[0]['B'])
        pos = graphviz_layout(self.lattice, prog='dot')

        plt.figure(figsize=(12, 8))

        # Отрисовка узлов
        nx.draw_networkx_nodes(self.lattice, pos, node_color="dodgerblue", node_shape="o")

        # Отрисовка ребер графа
        nx.draw_networkx_edges(self.lattice, pos, edge_color="turquoise", arrows=False, alpha=0.5)

        # Отрисовка подписей узлов
        node_labels = {
            i: '\n'.join(textwrap.wrap(
                f"{','.join(str(s) for s in self.concepts[i]['B'])}\n{','.join(str(s) for s in self.concepts[i]['A'])}",
                width=25  # Максимальная ширина строки
            ))
            for i in self.lattice.nodes()
        }
        nx.draw_networkx_labels(
            self.lattice,
            pos,
            labels=node_labels,
            font_color="black",
            font_size=8,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', pad=0.5)
        )

        plt.axis("off")

        plt.show()
        plt.pause(1)

    def find_element(self, element: str, concepts: list[tuple[int, dict]]):

        start_time = time.time()
        element = element.strip().lower()

        if element.startswith("f"):
            target_type = "A"
        elif element.startswith("d"):
            target_type = "B"
        else:
            print("⚠️ Неизвестный тип элемента:", element)
            return [], 0.0

        matching_concepts = [(i, concept) for i, concept in concepts if element in concept[target_type]]

        elapsed = time.time() - start_time
        print(f"⏱ Выполнение заняло {elapsed:.8f} секунд.")
        return matching_concepts, elapsed

    def interactive_multi_derivation_loop(self):
        current_table = self.context.copy()  # начальная таблица
        total_time = 0.0  # общее время
        used_f = set()
        used_d = set()

        while True:
            print("\n📊 Текущая таблица:")
            print(current_table)

            # Определение доступных f и d
            available_f = set(current_table.index) - used_f
            available_d = set(current_table.columns) - used_d

            if not available_f and not available_d:
                print("❌ Больше нет доступных элементов для ввода.")
                break

            print("\nДоступные элементы:")
            print(f"🔸 Объекты (f): {', '.join(sorted(available_f)) if available_f else '(пусто)'}")
            print(f"🔹 Признаки (d): {', '.join(sorted(available_d)) if available_d else '(пусто)'}")

            user_input = input("Введите элемент (например: f1 или d2), или 'q' для выхода: ").strip().lower()
            if user_input == 'q':
                break

            if user_input in used_f or user_input in used_d:
                print("⚠️ Этот элемент уже был использован. Введите другой.")
                continue

            if user_input.startswith("f") and user_input not in available_f:
                print("⚠️ Элемент не найден среди доступных f.")
                continue
            elif user_input.startswith("d") and user_input not in available_d:
                print("⚠️ Элемент не найден среди доступных d.")
                continue

            # Выполнение деривации
            current_table, elapsed = self.multi_derivation_procedure(user_input, current_table)
            total_time += elapsed

            # Запись использованного элемента
            if user_input.startswith("f"):
                used_f.add(user_input)
            elif user_input.startswith("d"):
                used_d.add(user_input)

            if current_table.empty:
                print("❌ Результат пуст. Таблица больше не содержит данных.")
                break

            print(f"\n🧮 Общее время всех шагов: {total_time:.8f} секунд.")

    def interactive_find_element_loop(self):
        import copy

        total_time = 0.0
        current_concepts = None  # None — значит искать по всем
        used_f = set()
        used_d = set()


        # Локальная копия концептов
        concepts_base = copy.deepcopy(self.concepts)

        while True:


            # Определяем пространство поиска
            if current_concepts is None:
                search_space = list(enumerate(concepts_base))
            else:
                search_space = current_concepts

            # Сбор доступных f и d (исключая уже использованные)
            available_f = set()
            available_d = set()
            for _, concept in search_space:
                available_f.update(concept['A'])
                available_d.update(concept['B'])

            available_f -= used_f
            available_d -= used_d

            if not available_f and not available_d:
                print("❌ Нет доступных элементов для поиска. Завершение.")
                break

            print(f"Доступные f: {', '.join(sorted(available_f)) if available_f else '(пусто)'}")
            print(f"Доступные d: {', '.join(sorted(available_d)) if available_d else '(пусто)'}")

            user_input = input("Введите элемент (например: f1 или d2), или 'q' для выхода: ").strip().lower()
            if user_input == 'q':
                break

            if user_input in used_f or user_input in used_d:
                print("⚠️ Этот элемент уже был использован. Введите другой.")
                continue

            if user_input.startswith("f") and user_input not in available_f:
                print("⚠️ Элемент не найден среди доступных f.")
                continue
            elif user_input.startswith("d") and user_input not in available_d:
                print("⚠️ Элемент не найден среди доступных d.")
                continue

            result, elapsed = self.find_element(user_input, search_space)
            total_time += elapsed

            if result:
                print(f"\n✅ Найдено {len(result)} концептов за {elapsed:.4f} секунд.")
                for i, c in result:
                    print(f"Концепт {i}: A = {c['A']}, B = {c['B']}")

                current_concepts = result

                # Обновление использованных элементов
                if user_input.startswith("f"):
                    used_f.add(user_input)
                elif user_input.startswith("d"):
                    used_d.add(user_input)
            else:
                print("❌ Концептов не найдено.")
                break

            print(f"🧮 Общее время всех шагов: {total_time:.8f} секунд.")

    def generate_auto_requests(self, count):
        total_time = 0.0
        all_requests = []
        unique_sets = set()
        max_attempts = count * 10  # Ограничим число попыток, чтобы избежать бесконечного цикла

        attempts = 0
        while len(all_requests) < count and attempts < max_attempts:
            attempts += 1
            used_f = set()
            used_d = set()
            current_table = self.context.copy()
            request = []
            max_length = random.randint(1, 7)

            while len(request) < max_length:
                if (current_table.values == 1).all():
                    break

                available_f = list(set(current_table.index) - used_f)
                available_d = list(set(current_table.columns) - used_d)

                candidates = []
                if available_f:
                    candidates.append('f')
                if available_d:
                    candidates.append('d')

                if not candidates:
                    break

                choice_type = random.choice(candidates)
                if choice_type == 'f':
                    selected = random.choice(available_f)
                    used_f.add(selected)
                else:
                    selected = random.choice(available_d)
                    used_d.add(selected)

                current_table, elapsed = self.multi_derivation_procedure(selected, current_table)
                total_time += elapsed
                request.append(selected)

            if request:
                request_key = frozenset(request)
                if request_key not in unique_sets:
                    unique_sets.add(request_key)
                    all_requests.append(request)
            else:
                print("⚠️ Запрос оказался пустым — возможно, ошибка в логике.")

        if len(all_requests) < count:
            print(
                f"⚠️ Удалось сгенерировать только {len(all_requests)} уникальных запросов из {count} после {attempts} попыток.")
        else:
            print(
                f"\n✅ Генерация {len(all_requests)} уникальных наборов завершена. Общее время: {total_time:.8f} секунд.")

        return all_requests

    def user_interface(self):
        while True:
            print("\nВыберите действие:")
            print("1. Выполнить поиск через деревацию")
            print("2. Выполнить поиск через концепты")
            print("3. Найти достижимые концепты от супремума/инфимума")
            print("4. Автоматическая генерация запросов")
            print("5. Построить граф решётки")
            print("6. Выход")

            choice = input("Введите номер выбранной опции (1-6): ")

            if choice == "1":
                self.interactive_multi_derivation_loop()
            elif choice == "2":
                self.interactive_find_element_loop()
            elif choice == "3":
                self.find_reachable_concepts()
            elif choice == "4":
                n = int(input("Введите количество запросов для генерации: "))
                self.generated_requests = self.generate_auto_requests(n)
                print("\n📋 Сгенерированные запросы:")
                for i, req in enumerate(self.generated_requests, 1):
                    print(f"{i}: {req}")
            elif choice == "5":
                self.fill_lattice()
                self.lat_draw()
            elif choice == "6":
                print("Выход из программы...")
                break
            else:
                print("Неверный выбор. Пожалуйста, попробуйте снова.")

    def lattice_query_support(self, axis, el, bound_n):
        if axis == 'A':
            if el in lat.concepts[bound_n]['A']:
                return bound_n
            else:
                items_list = list(lat.lattice.pred[bound_n].items())
                for n in range(len(list(self.lattice.pred[bound_n].items()))):
                    if el in list(self.lattice.pred[bound_n].items())[n][1]['add_m'].split(','):
                        return list(self.lattice.pred[bound_n].items())[n][0]
        elif axis == 'B':
            if el in lat.concepts[bound_n]['B']:
                return bound_n
            else:
                items_list = list(lat.lattice.succ[bound_n].items())
                for n in range(len(list(self.lattice.succ[bound_n].items()))):
                    if el in list(self.lattice.succ[bound_n].items())[n][1]['add_d'].split(','):
                        return list(self.lattice.succ[bound_n].items())[n][0]
        else:
            return 0

class MockContext:
    def __init__(self, num_rows: int, num_cols: int, density: float, seed: int = None):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.density = max(0.0, min(1.0, density))
        self.seed = seed
        self.context_df = None
        self.generate_context()

    @property
    def shape_ratio(self) -> float:
        return round(min(self.num_rows, self.num_cols) / max(self.num_rows, self.num_cols), 3)

    @property
    def actual_density(self) -> float:
        if self.context_df is not None:
            total = self.num_rows * self.num_cols
            ones = self.context_df.values.sum()
            return round(ones / total, 3)
        return 0.0

    def generate_context(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        data = np.random.rand(self.num_rows, self.num_cols) < self.density
        self.context_df = pd.DataFrame(
            data.astype(int),
            index=[f"f{i+1}" for i in range(self.num_rows)],
            columns=[f"d{j+1}" for j in range(self.num_cols)]
        )

    def get_parameters(self):
        return {
            "Количество строк: ": self.num_rows,
            "Количество столбцов: ": self.num_cols,
            "Отношение строки/столбцы: ": self.shape_ratio,
            "Введеная разряженость: ": self.density,
            "Реальная разряженость: ": self.actual_density,
            "Seed: ":self.seed
        }

    def save_to_csv(self, path: str = "mock_out.csv"):
        """
        Сохраняет DataFrame в CSV-файл, автоматически генерируя уникальное имя файла
        с указанием seed, плотности и размерности.
        """
        if self.context_df is not None:
            now = datetime.datetime.now()
            timestamp = now.strftime("%d-%m-%Y_%H-%M-%S")
            base_name = path.rsplit('.', 1)[0] if '.' in path else path
            file_extension = '.' + path.rsplit('.', 1)[1] if '.' in path else '.csv'

            seed_part = f"_seed{self.seed}" if self.seed is not None else ""
            density_part = f"_density{self.density}"
            size_part = f"_{self.num_rows}x{self.num_cols}"

            unique_path = f"{base_name}_{timestamp}{seed_part}{density_part}{size_part}{file_extension}"

            self.context_df.to_csv(unique_path)
            print(f"Таблица сохранена в файл: {unique_path}")
        else:
            print("DataFrame отсутствует. Сохранение невозможно.")

    @classmethod
    def from_user_input(cls):
        try:
            rows = int(input("Введите количество строк: "))
            cols = int(input("Введите количество столбцов: "))
            density_input = input("Введите разреженность матрицы (0.0 — 1.0): ").replace(',', '.')
            density = float(density_input)
            seed = input("Введите seed для генерации (или нажмите Enter): ")
            seed = int(seed) if seed else None
            return cls(rows, cols, density, seed)
        except ValueError:
            print("Ошибка ввода! Попробуйте снова.")
            return cls.from_user_input()
        


if __name__ == '__main__':

    mock = MockContext.from_user_input()
    print("\nПараметры сгенерированного контекста:")
    for k, v in mock.get_parameters().items():
        print(f"{k}: {v}")


    mock.save_to_csv("table.csv")



    table = mock.context_df

    print("\nИнициализация fca_lattice...")
    start_time = time.time()
    lat = fca_lattice(table)
    print("Загрузка --- %.2f секунд ---" % (time.time() - start_time))


    print("\nГенерация концептов...")
    start_time = time.time()
    lat.in_close(0, 0, 0)
    print("Генерация завершена за %.2f секунд" % (time.time() - start_time))
    print("Количество концептов:", len(lat.concepts))


    print("\nСписок всех объектов и признаков:")
    lat.print_indexes()





#Комментарий для проверки