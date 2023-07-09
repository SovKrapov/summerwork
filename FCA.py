import time
import textwrap

import joblib

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
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
        """
        Проверка концепта на каноничность. Классический алгоритм
        :param column: номер столца на основе которого сгенерирован концепт
        :param new_a: объем нового концепта, который нужно проверить на каноничность
        :param r: номер концепта на основе которго сгененирован новый концепт
        :return: результат проверки
        """
        for i in range(column, -1, -1):
            if self.context.columns[i] not in self.concepts[r]['B']:
                if new_a.issubset(self.context_derivation_1.iloc[i]):
                    return False
        return True

    def in_close(self, column: int, r: int, threshold=0.0):
        """
        Закрывает концепт по контексту. Классический алгоритм
        :param column: номер столбца с которого начать проверку
        :param r: номер текущего контекста
        :param threshold: порог (использовался при расчете концептов заданного объема)
        :return:
        """
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


    def derivation(self, q_val: str, axis=0):
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

    def find_element(lat):
        set_type = input("Введите тип множества (F или D): ")
        element = input("Введите элемент: ")

        if set_type.upper() == 'F':
            axis1 = 'A'
        elif set_type.upper() == 'D':
            axis1 = 'B'
        else:
            print("Недопустимый тип множества.")
            return

        matching_concepts = []
        for i in range(len(lat.concepts)):
            if element in lat.concepts[i][axis1]:
                matching_concepts.append(i)

        if matching_concepts:
            print("Концепты, содержащие элемент:")
            for concept_idx in matching_concepts:
                concept = lat.concepts[concept_idx]
                print(f"Концепт {concept_idx}: A = {concept['A']}, B = {concept['B']}")
        else:
            print("No concepts found containing the element.")

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
        
if __name__ == '__main__':
    table = pd.read_csv("out.csv", header=0,index_col=0)
    #binary = pd.read_csv('order_products__prior.csv', header=0)
    #table = pd.pivot_table(binary.head(10046), values='add_to_cart_order', index=['order_id'], columns=['product_id'],aggfunc=np.count_nonzero, fill_value=0)
    #запуск таймера
    start_time=time.time()
    #   Инициализация объекта
    lat = fca_lattice(table)
    print("Загрузка --- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
#     Вызов процедуры расчета решетки. in_close - классический расчет для небольших контекстов, 
#     stack_my_close - пошаговый расчет (считает только одну часть концептов)
    lat.in_close(0, 0, 0)

    print("Генерация концептов --- %s seconds ---" % (time.time() - start_time))
    print(len(lat.concepts))

#Комментарий для проверки