import numpy as np
import random, pickle
from tqdm import tqdm
import cv2, time, argparse

# 돌연변이 생성 비율
MUTATION_PROB = 0.01

# 각 세대는 100개의 해로 구성되도록 수정
# 각 해는 100개의 원 또는 폴리곤으로 구성되어 있다.
# 이때 100의 중심점 반지름 등을 전부 기억해서 100개의 도형이 유지되도록 하는 것이 목적
# 적합도는 원본 사진과의 디스턴스로 구할 예정이고
# Xover 방식은 미정
# mutaion은 도형 및 색상 대체.

#commit test

class chromoSome:
    def __init__(self, real_img, max_shapes=100, chromo_data=None, shapes_list =[]):
        self.real_img = real_img
        self.img_size = real_img.shape
        self.max_shapes = max_shapes
        self.shapes_list = shapes_list

        if chromo_data is None:
            self.create_random_img()
        else:
            self.img = chromo_data

    def __repr__(self):
        return "chromosome fitenss : {}".format(self.fitness)

    @property
    def fitness(self):
        score = 0
        dist = np.linalg.norm(self.real_img.astype('float') - self.img.astype('float')) / (self.img_size[0] * self.img_size[1])

        # 높을수록 좋음
        score = 1. / dist

        return score

    def create_random_img(self):
        #초기해 생성을 위한 코드
        self.img = np.zeros(self.img_size, np.uint8)
        for _ in range(self.max_shapes):
            random_assign = random.choice([chromoSome.assign_circle, chromoSome.assign_polygon])
            self.shapes_list.append(random_assign(self.img, self.max_shapes, self.img_size))

    def assign_circle(img, max_shapes, img_size):
        #초기해일 경우에 빈 프레임 들어옴

        overlay  = img.copy()
        #mutation으로 추가할 도형 개수

        center_x = np.random.randint(0, img_size[1])
        center_y = np.random.randint(0, img_size[0])
        radius = np.random.randint(0, img_size[0]/4)
        #radius   = np.random.randint(0, int(img_size[0] / (1.1*res)))
        opacity  = np.random.rand(1)[0]
        color    = chromoSome.get_bgr_color()
        cv2.circle(overlay, (center_x, center_y), radius, color, -1) #img.copy에 중심x 중심y 반지름 색 그림.
        cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)  # img에 opacity, 1-opacity 만큼의 비율로 합쳐서 반환

        return ["circle", None, center_x, center_y, radius, opacity, color] #원의 중심 좌표, 반지름, 투명도, 색상 반환

    def assign_polygon(img, max_shapes, img_size):
        pts = []
        point = [np.random.randint(0, img_size[1]), np.random.randint(0, img_size[0])]
        pts.append(point)

        n_shapes = np.random.randint(0, max_shapes)

        for i in range(n_shapes):
            new_point = [point[0] + np.random.randint(-1, 2) * np.random.randint(0, int(img_size[1] / 4)), point[1] + np.random.randint(-1, 2) * np.random.randint(0, int(img_size[0] / 4))]
            pts.append(new_point)

        pts 	 = np.array(pts)
        pts 	 = pts.reshape((-1, 1, 2))
        opacity  = np.random.rand(1)[0]
        color    = chromoSome.get_bgr_color()

        overlay  = img.copy()
        cv2.fillPoly(overlay, [pts], color, 8)
        cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)

        return ["polygon", pts, None, None, None, opacity, color]

    def get_bgr_color():
        blue  = np.random.randint(0, 255)
        green = np.random.randint(0, 255)
        red   = np.random.randint(0, 255)
        return (blue, green, red)


class Generation:
    cnt = 0
    def __init__(self, population):
        Generation.cnt += 1
        self.generation_lv = Generation.cnt
        self.population = population
        self.sorted_pop = self.sort_pop()

    def __repr__(self):
        return "Generation Level : {}".format(self.generation_lv)

    def rangeArray(self, range_array):
        k = 3
        k_related_fitness = [(((self.sorted_pop[-1].fitness - self.sorted_pop[i].fitness) + (self.sorted_pop[-1].fitness - self.sorted_pop[0].fitness)) / (k-1)) for i in range(len(self.sorted_pop))]
        range_array[0] = k_related_fitness[0]
        for i in range(1, len(self.population)):
            range_array[i] = range_array[i-1] + k_related_fitness[i]
        return range_array

    def sampling(self):
        temp_array = self.rangeArray([ 0 for _ in range(len(self.population))])
        selection_index = []
        while len(selection_index) != 2: # 부모해 2개 고르는 과정

            # value = rand()/(float)RAND_MAX*temp_array[-1]
            value = random.uniform(0, temp_array[-1])

            for i in range(len(temp_array)):
                if value <= temp_array[i]:
                    if i not in selection_index:
                        selection_index.append(i)
                        break
        return selection_index[0], selection_index[1]

    def selection(self):
        p1, p2 = self.sampling()
        p
        print(p1,p2)
        self.best_chromo, self.second_chromo = self.sorted_pop[p1], self.sorted_pop[p2]
        # self.best_chromo = self.sorted_pop[0]
        # self.second_chromo = self.sorted_pop[1]

    def evolution(self):
        # 최상위 Fitness 유전자 설정
        # 1위, 2위 개체 설정

        # 돌연변이 갯수 설정
        n_population = len(self.population)
        # 100 * 0.01 = 1 = mutation_cnt
        mutation_cnt = n_population * MUTATION_PROB

        children = list()
        for num in range(n_population):
            print(num, end = " ")
            #selection 함수 호출해서 전역 변수에 부모 두개 고름
            self.selection()
            print(self.best_chromo, self.second_chromo)
            child = self.make_child(num)

            children.append(child)

        return Generation(children)

    def sort_pop(self):
        # 유전자 Fitness 좋은 순으로 정렬
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        return sorted_pop

    def drawChild(self, child_shapes_list, new_img):
        for child_shape in child_shapes_list:
            if child_shape[0] == "circle":
                center_x = child_shape[2]
                center_y = child_shape[3]
                radius = child_shape[4]
                opacity = child_shape[5]
                color = child_shape[6]
                cv2.circle(new_img, (center_x, center_y), radius, color, -1)

            elif child_shape[0] == "polygon":
                pts = child_shape[1]
                opacity = child_shape[5]
                color = child_shape[6]
                cv2.fillPoly(new_img, [pts], color, 8)

    def one_point_xover(self, new_img):
        random_point = np.random.randint(1, 99)
        child_shapes_list = []
        for temp_shapes in self.best_chromo.shapes_list[0:random_point]:
            child_shapes_list.append(temp_shapes)
        for temp_shapes in self.best_chromo.shapes_list[random_point:100]:
            child_shapes_list.append(temp_shapes)
        return child_shapes_list

    def make_child(self, num):#Xover 방식인데, 부모 두개를 임의의 투명도를 설정하여 합침.
        # 랜덤으로 첫번째 opacity(불투명) 비중설정
        ind1_weight = np.random.rand(1)[0]
        new_image = np.zeros((self.population[0].img_size), dtype=np.uint8)
        #self.best_chormo.img와 self.second_chromo.img 의 도형을 크로스 오버해야 함.
        child_shapes_list = self.one_point_xover(new_image)
        # #addWeighted(이미지1, 이미지1의 투명도, 이미지2, 1-이미지1의 투명도, 저장 대상)
        # cv2.addWeighted(self.best_chromo.img, ind1_weight, self.second_chromo.img, 1 - ind1_weight, 0, new_image)

        #mutation 넣고 최종적으로 그리기
        if np.random.randint(0, 101) == 5: # mutation 확률 0.01
            child_shapes_list[np.random.randint(0,100)] = self.make_mutation(child_shapes_list)

        #그려야함
        self.drawChild(child_shapes_list, new_image) #결과로 new_image에 child_shape_list 항목 그림.

        cv2.imwrite("./training_picaso/"+str(self.generation_lv)+"_"+str(num)+"_img.jpg", new_image)

        child = chromoSome(real_img=self.best_chromo.real_img, chromo_data=new_image, shapes_list=child_shapes_list)

        return child

    def make_mutation(self, child_shapes_list):
        print("mutation 발생")
        if np.random.randint(0, 2) == 0:
            center_x = np.random.randint(0, self.population[0].img_size[1])
            center_y = np.random.randint(0, self.population[0].img_size[0])
            radius = np.random.randint(0, self.population[0].img_size[0]/4)
            opacity  = np.random.rand(1)[0]
            color    = chromoSome.get_bgr_color()
            return ["circle", None, center_x, center_y, radius, opacity, color] #원의 중심 좌표, 반지름, 투명도, 색상 반환

        else:
            pts = []
            point = [np.random.randint(0, self.population[0].img_size[1]), np.random.randint(0, self.population[0].img_size[0])]
            pts.append(point)

            n_shapes = np.random.randint(0, self.population[0].max_shapes)

            for i in range(n_shapes):
                new_point = [point[0] + np.random.randint(-1, 2) * np.random.randint(0, int(img_size[1] / 4)), point[1] + np.random.randint(-1, 2) * np.random.randint(0, int(img_size[0] / 4))]
                pts.append(new_point)

            pts 	 = np.array(pts)
            pts 	 = pts.reshape((-1, 1, 2))
            opacity  = np.random.rand(1)[0]
            color    = chromoSome.get_bgr_color()
            return ["polygon", pts, None, None, None, opacity, color]

    @property
    def mean_fitness(self):
        return np.mean([chromosome.fitness for chromosome in self.population])

    @property
    def get_best(self):
        return sorted(self.population, key=lambda x: x.fitness, reverse=True)[0]

    def fitness(self):
        return np.mean([chromo.fitness for chromo in self.population])


def save_pickle(data, filename):
    with open("{}.pickle".format(filename), 'wb') as t:
        pickle.dump(data, t)
    print("PICKLE SAVE DONE")

def load_pickle(filepath):
    with open("{}.pickle".format(filepath), 'rb') as t:
        data = pickle.load(t)
    print("PICKLE LOAD DONE")
    return data


def main(file_path, n_population, n_generation):
    #원본 이미지 읽음
    IMA_ARR = cv2.imread(file_path)
    #적합도 리스트 생성
    fitness_list = list()
    #초기해 생성 n_population 만큼 생성하고, choromSome 인스턴스 생성
    initial_pop = [chromoSome(real_img=IMA_ARR) for _ in range(n_population)]

    gen = Generation(initial_pop)

    #세대 수 : 10,000, 개체 수: 100

    try:
        for i in tqdm(range(n_generation)):
            gen = gen.evolution()
            best = gen.get_best
            fitness_list.append(best.fitness)

            cv2.imwrite("./img_picaso/{}_img.jpg".format(i), best.img)
    finally:
        save_pickle(fitness_list, "./pickle_picaso/fitness_list")
        save_pickle(gen, "./pickle_picaso/last_generation")

if __name__ == "__main__":

    DEBUG = False
    if DEBUG:
        file_path = "picaso.png"
        n_population = 100
        n_generation = 100000
        main(file_path, n_population, n_generation)

    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("-np", "--n_pop", required=False, type=int)
        ap.add_argument("-ng", "--n_gen", required=False, type=int)
        ap.add_argument("-path", "--file_path", required=True, type=str)

        args = vars(ap.parse_args())

        if args["n_pop"] != None:
            n_population = args["n_pop"]

        if args['n_gen'] != None:
            n_generation = args['n_gen']

        if args['file_path'] != None:
            file_path = args['file_path']

        main(file_path, n_population, n_generation)
