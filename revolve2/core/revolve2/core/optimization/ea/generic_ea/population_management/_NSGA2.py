import copy
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import cm
import os

class NSGA2:
    def __init__(self, path_to_dir: str, debug: bool = True):
        self.path_to_dir = path_to_dir
        self.debug = debug

    def __call__(self, population_individuals,generation_index):
        population_size = len(population_individuals)

        # Preparate the objectives as a matrix of individuals in the rows and fitnesses in the columns.
        objectives = np.zeros((population_size , max(1, len(population_individuals[0].objectives))))  # TODO fitnesses is 0

        # Fill the objectives with all individual from the population
        all_individuals = copy.deepcopy(population_individuals)


        print("creating objectives")
        for index, individual in enumerate(all_individuals):
            # Negative fitness due to minimum search, TODO can be changed to be a default maximization NSGA.
            objectives[index, :] = [-objective for objective in individual.objectives]

        print("starting sorting")
        # Perform the NSGA Algorithm
        front_no, max_front = self.nd_sort(objectives, np.inf)
        crowd_dis = self.crowding_distance(objectives, front_no)
        sorted_fronts = self.sort_fronts(objectives, front_no, crowd_dis)

        print("creating new individuals")
        # Select the all the individuals for ranking
        new_individuals = [all_individuals[index] for index in sorted_fronts]
        discarded_individuals = []

        if self.debug:
            self._visualize(objectives, sorted_fronts, new_individuals, discarded_individuals, front_no, generation_index)

        return new_individuals

    def _visualize(self, objectives, sorted_fronts, new_individuals, discarded_population, front_no, generation_index):
        number_of_fronts = int(max(front_no))
        colors = cm.rainbow(np.linspace(1, 0, number_of_fronts))

        if objectives.shape[1] == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        fronts = {}

        for individual_index in sorted_fronts:
            front_number = int(front_no[individual_index]) - 1

            if front_number in fronts.keys():
                points_in_front = fronts.get(front_number)
                points_in_front.append((-objectives[individual_index, 0], -objectives[individual_index, 1]))
                fronts[front_number] = points_in_front
            else:
                fronts[front_number] = [(-objectives[individual_index, 0], -objectives[individual_index, 1])]

            if objectives.shape[1] == 3:
                ax.scatter(objectives[individual_index, 0], objectives[individual_index, 1], objectives[individual_index, 2], s=10*(number_of_fronts-front_number),
                           color=colors[front_number])
            else:
                ax.scatter(-objectives[individual_index, 0], -objectives[individual_index, 1], s=5*(number_of_fronts-front_number),
                            color=colors[front_number])

        for (front_number, points_in_front) in fronts.items():
            sorted_front = sorted(points_in_front, key=(lambda a: a[0]))
            ax.plot([point[0] for point in sorted_front], [point[1] for point in sorted_front], color=colors[front_number])

        for individual in new_individuals:
            ax.scatter(individual.objectives[0], individual.objectives[1], s=3, color='black')

        for individual in discarded_population:
            ax.scatter(individual.objectives[0], individual.objectives[1], s=2, color='white')

        # print(str(generation_index))
        # print("Fronts plot" + str(generation_index) + ".png")
        fig.savefig(self.path_to_dir + "/Fronts plot" + str(generation_index) + ".png")
        del ax

    def nd_sort(self, objectives, max_range):
        number_of_individuals, number_of_objectives = objectives.shape
        sorted_matrix = np.lexsort(objectives[:,::-1].T)  # loc1 is the position of the new matrix element in the old matrix, sorted from the first column in order
        sorted_objectives = objectives[sorted_matrix]
        inverse_sorted_indexes = sorted_matrix.argsort()  # loc2 is the position of the old matrix element in the new matrix
        frontno = np.ones(number_of_individuals) * (np.inf)  # Initialize all levels to np.inf
        maxfno = 0  # 0
        while (np.sum(frontno < np.inf) < min(max_range, number_of_individuals)):  # The number of individuals assigned to the rank does not exceed the number of individuals to be sorted
            maxfno = maxfno + 1
            for i in range(number_of_individuals):
                if (frontno[i] == np.inf):
                    dominated = 0
                    for j in range(i):
                        if (frontno[j] == maxfno):
                            m = 0
                            flag = 0
                            while (m < number_of_objectives and sorted_objectives[i, m] >= sorted_objectives[j, m]):
                                if (sorted_objectives[i, m] == sorted_objectives[j, m]):  # does not constitute a dominant relationship
                                    flag = flag + 1
                                m = m + 1
                            if (m >= number_of_objectives and flag < number_of_objectives):
                                dominated = 1
                                break
                    if dominated == 0:
                        frontno[i] = maxfno
        frontno = frontno[inverse_sorted_indexes]
        return frontno, maxfno

    def crowding_distance(self, objectives, front_number):
        """
        The crowding distance of each Pareto front
        :param objectives: objective vectors
        :param front_number: front numbers
        :return: crowding distance
        """
        number_of_individuals, number_of_objectives = np.shape(objectives)  # todo x, y?
        crowd_dis = np.zeros(number_of_individuals)

        # Initialize fronts
        front = np.unique(front_number)
        fronts = front[front != np.inf]
        for f in range(len(fronts)):
            front = np.array([k for k in range(len(front_number)) if front_number[k] == fronts[f]])

            # Find bounds
            Fmax = objectives[front, :].max(0)
            Fmin = objectives[front, :].min(0)

            # For each objective sort the front
            for i in range(number_of_objectives):
                rank = np.argsort(objectives[front, i])

                # Initialize Crowding distance
                crowd_dis[front[rank[0]]] = np.inf
                crowd_dis[front[rank[-1]]] = np.inf

                for j in range(1, len(front) - 1):
                    crowd_dis[front[rank[j]]] += (objectives[(front[rank[j + 1]], i)] -
                                                  objectives[(front[rank[j - 1]], i)]) / (Fmax[i] - Fmin[i])
        return crowd_dis

    def sort_fronts(self, objectives, front_no, crowd_dis):
        front_dict = dict() # dictionary indexed by front number inserting objective with crowd distance as tuple

        for objective_index in range(len(objectives)):
            if front_no[objective_index] not in front_dict.keys():
                front_dict[front_no[objective_index]] = [(crowd_dis[objective_index], objective_index)]
            else:
                front_dict[front_no[objective_index]].append((crowd_dis[objective_index], objective_index))

        sorted_fronts = []
        sorted_keys = sorted(front_dict.keys())

        for key in sorted_keys:
            front_dict[key].sort(key=lambda x: x[0], reverse=True)
            for element in front_dict[key]:
                sorted_fronts.append(element[1])

        return sorted_fronts

