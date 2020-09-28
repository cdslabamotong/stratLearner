from base import StructuredModel
import numpy as np
import sys
import heapq
import time
import random
import math
import multiprocessing
import copy


class Utils(object):
    def greeting(name):
        print("Hello, " + name)
    
    def getData(self,path, Num):
        file1 = open(path, 'r') 
        lineNum = 1
        X = []
        Y = []
        X_influence = []
        Y_influence = []
        while True: 
            line = file1.readline() 
            if not line: 
                break 
            seedset = set(line.split())
            if len(Y) < Num:
                if lineNum % 5 == 1:
                    X.append(seedset)
                if lineNum % 5 == 2:
                    Y.append(seedset)
                if lineNum % 5 == 3:
                    X_influence.append(float(line))
                if lineNum % 5 == 4:
                    Y_influence.append(float(line))
            lineNum += 1
        if (len(X) != Num) or (len(Y) != Num):
            sys.exit("getData: data fetch failed with sizes: {} {}".format(
                len(X),len(Y))) 
        return X, Y, X_influence, Y_influence
    
    def getDataTrainTest(self, path, trainNum, testNum):
        file1 = open(path, 'r') 
        lineNum = 1
        X_train = []
        Y_train = []
        X_train_influence = []
        Y_train_influence = []
        
        X_test = []
        Y_test = []
        X_test_influence = []
        Y_test_influence = []
        while True: 
            line = file1.readline() 
            if not line: 
                break 
            seedset = set(line.split())
            if len(Y_train) < trainNum:
                if lineNum % 5 == 1:
                    X_train.append(seedset)
                if lineNum % 5 == 2:
                    Y_train.append(seedset)
                if lineNum % 5 == 3:
                    X_train_influence.append(float(line))
                if lineNum % 5 == 4:
                    Y_train_influence.append(float(line))
            else:
                if len(Y_test) < testNum:
                    if lineNum % 5 == 1:
                        X_test.append(seedset)
                    if lineNum % 5 == 2:
                        Y_test.append(seedset)
                    if lineNum % 5 == 3:
                        X_test_influence.append(float(line))
                    if lineNum % 5 == 4:
                        Y_test_influence.append(float(line))
                        
            lineNum += 1
        if (len(X_train) != trainNum) or (len(Y_test) != testNum):
            sys.exit("getData: data fetch failed with sizes: {} {}".format(
                len(X_train),len(Y_train))) 
        return X_train, Y_train, X_train_influence, Y_train_influence, X_test, Y_test, X_test_influence, Y_test_influence
    
    def getDataTrainTestRandom(self,path, trainNum, testNum, Max):
        lineNums=(np.random.permutation(Max)*5)[0:(trainNum+testNum)]      
        lineNums.sort()
        file1 = open(path, 'r') 
        lineNum = 0
        X_train, Y_train, X_train_influence, Y_train_influence = ([] for i in range(4))
        X_test, Y_test, X_test_influence, Y_test_influence= ([] for i in range(4))

        while len(lineNums)>0:
            line = file1.readline() 
            if not line: 
                break 
            if lineNum != lineNums[0]:
                lineNum += 1 
            else:
                if(len(Y_train)<trainNum):
                    seedset = set(line.split())
                    X_train.append(seedset)
                    lineNum += 1   
                    
                    line = file1.readline()
                    seedset = set(line.split())
                    Y_train.append(seedset)
                    lineNum += 1 
                    
                    line = file1.readline()
                    X_train_influence.append(float(line))
                    lineNum += 1
                    
                    line = file1.readline()
                    Y_train_influence.append(float(line))
                    lineNum += 1 
                    lineNums=np.delete(lineNums, 0)
                    #print(Y_train)
                    #print("train++", len(Y_train),len(lineNums))
                else:
                    seedset = set(line.split())
                    X_test.append(seedset)
                    lineNum += 1   
                    
                    line = file1.readline()
                    seedset = set(line.split())
                    Y_test.append(seedset)
                    lineNum += 1 
                    
                    line = file1.readline()
                    X_test_influence.append(float(line))
                    lineNum += 1
                    
                    line = file1.readline()
                    Y_test_influence.append(float(line))
                    lineNum += 1 
                    lineNums=np.delete(lineNums, 0)
                    #print("test++ {}"+format(len(lineNums)))
            
        if (len(X_train) != trainNum) or (len(Y_test) != testNum):
            sys.exit("getDataRandom: data fetch failed with sizes: X_train {} Y_test {}".format(
                len(X_train),len(Y_test))) 
        return X_train, Y_train, X_train_influence, Y_train_influence, X_test, Y_test, X_test_influence, Y_test_influence
    
    def getDataRandom(self,path, Num, Max):
        lineNums=(np.random.permutation(Max)*5)[0:Num]
        '''
        lineNums = []
        while len(lineNums)<Num:
            num = math.ceil(random.uniform(0, 1)*Max) 
            if 5*num not in lineNums:
                lineNums.append(5*num)
        '''
        
        lineNums.sort()
        #print(lineNums)
        file1 = open(path, 'r') 
        lineNum = 0
        X = []
        Y = []
        X_influence = []
        Y_influence = []
        while len(lineNums)>0:
            line = file1.readline() 
            if not line: 
                break 
            if lineNum != lineNums[0]:
                lineNum += 1 
            else:
                seedset = set(line.split())
                X.append(seedset)
                lineNum += 1   
                
                line = file1.readline()
                seedset = set(line.split())
                Y.append(seedset)
                lineNum += 1 
                
                line = file1.readline()
                X_influence.append(float(line))
                lineNum += 1
                
                line = file1.readline()
                Y_influence.append(float(line))
                lineNum += 1 
                lineNums=np.delete(lineNums, 0)                   
            
        if (len(X) != Num) or (len(Y) != Num):
            sys.exit("getDataRandom: data fetch failed with sizes: {} {}".format(
                len(X),len(Y))) 
        return X, Y, X_influence, Y_influence
    
    def testFunction(self, model, testNum,thread,X_test,Y_test, Y_pred, infTimes, random_pred = False):
        if random_pred :
            Y_pred =[]
            for x in X_test:   
                w=np.random.random((1, model.size_joint_feature))
                Y_pred.append(model.inference(x, w))
        
        
        
        block_size =int (testNum/thread);
        p = multiprocessing.Pool(thread)
        
        influence_Xs = p.starmap(model.instance.testInfluence_0_block, ((X_test[i*block_size:(i+1)*block_size], infTimes) for i in range(thread)))
        p.close()
        p.join()
        
        p = multiprocessing.Pool(thread)
        influence_Ys = p.starmap(model.instance.testInfluence_0_block, ((X_test[i*block_size:(i+1)*block_size], infTimes, Y_test[i*block_size:(i+1)*block_size]) for i in range(thread)))
        p.close()
        p.join()
        
        p = multiprocessing.Pool(thread)
        influence_Y_preds = p.starmap(model.instance.testInfluence_0_block, ((X_test[i*block_size:(i+1)*block_size], infTimes, Y_pred[i*block_size:(i+1)*block_size]) for i in range(thread)))
        p.close()
        p.join()
        
        
        influence_X=[]
        influence_Y=[]
        influence_Y_pred=[]
        for i in range(thread):
            influence_X.extend(influence_Xs[i])
            influence_Y.extend(influence_Ys[i])
            influence_Y_pred.extend(influence_Y_preds[i])
 
        
        reduce_percent_opt=[]
        reduce_percent_pre = []
        com_to_opt = []
        error_abs = []
        error_ratio = []
        for influence_x, influence_y, influence_y_pred in zip(influence_X, influence_Y, influence_Y_pred):
            #print("{} {} {} {} {}".format(influence_x,influence_y,influence_y_pred, influence_x_read, influence_y_read))
            reduce_percent_opt.append((influence_x-influence_y)/influence_x)
            reduce_percent_pre.append( (influence_x-influence_y_pred)/influence_x)
            com_to_opt.append((influence_x-influence_y_pred)/(influence_x-influence_y+0.01))
            error_abs.append((influence_y_pred-influence_y))
            error_ratio.append((influence_y_pred-influence_y)/influence_y)
            #print()
        
        print("error_abs: {} +- {}".format(np.mean(np.array(error_abs)), np.std(np.array(error_abs))))
        print("com_to_opt: {} +- {}".format(np.mean(np.array(com_to_opt)), np.std(np.array(com_to_opt))))
        
    def pooling(self,one_slack_svm, ratio):
        maxWeight=max(one_slack_svm.w)
        indexList=[]
        for i in range(len(one_slack_svm.w)):
            if one_slack_svm.w[i]>ratio*maxWeight:
                indexList.append(i)
        new_diffusionGraphs=[]
        for i in indexList:
            new_diffusionGraphs.append(copy.deepcopy(one_slack_svm.model.instance.diffusionGraphs[i]))
        one_slack_svm.model.instance.diffusionGraphs=new_diffusionGraphs
        one_slack_svm.model.instance.featureNum=len(indexList)
        one_slack_svm.model.size_joint_feature=len(indexList)
        
    
class Train(object):
    def __init__(self, attack, protect, a_influence, p_influence):
        self.attack=attack
        self.protect=protect
        self.a_influence=a_influence
        self.p_influence=p_influence

class SocialGraph(object):
    
    class Node(object):
        def __init__(self,index):
            self.index = index
            self.neighbor = {}
            self.in_degree = 0
            self.out_degree = 0
        def print(self):
            print(self.index)
            for node in self.neighbor:
                print("{} {} {} {}".format(str(self.index), str(node)
                                                 , str(self.neighbor[node][0]), str(self.neighbor[node][1])))
                
    def __init__(self, path, vNum):
        self.nodes={}
        self.vNum = vNum
        
        for v in range(self.vNum):
             node = self.Node(str(v))
             node.neighbor={}
             self.nodes[str(v)]=node
             
        file1 = open(path, 'r') 
        while True: 
            line = file1.readline() 
            if not line: 
                break          
            ints = line.split()
            node1 = ints[0]
            node2 = ints[1]
            para_1 = float(ints[2])
            para_2 = float(ints[3])
            
            if node1 in self.nodes:
                self.nodes[node1].neighbor[node2]=[para_1, para_2]
                self.nodes[node1].out_degree += 1
                self.nodes[node2].in_degree += 1
            else:
                sys.exit("non existing node") 
                
            if node2 not in self.nodes:
                sys.exit("non existing node") 
                
        #print(path + " read") 
        
    def print(self):
        for node in self.nodes:
            self.nodes[node].print()
    
    def getNeighborsByHot(self, y, hotNum):
        temp = y.copy()
        neighbors = y.copy()
        for _ in range(hotNum):
            for current in neighbors:
                for current_to in self.nodes[current].neighbor:   
                    temp.add(current_to)
            neighbors = temp.copy()
        return neighbors
    
    def spreadMulti_n0(self,x,y,times):
        return self.vNum-self.spreadMulti(x,y,times)
    
    def spreadMulti(self, x,y,times):
        local_state = random.Random()
        count = 0
        for _ in range(times):
            count += self.spreadOnce(x,y,local_state)
        return count/times
            
    def spreadMulti_P(self, x,y,times, thread):
        if not isinstance(thread, int):
            sys.exit("thread should be int") 
        if thread >1:
            p = multiprocessing.Pool(thread)
            counts = sum(p.starmap(self.spreadMulti, ((x,y,int(times/thread))for _ in range(thread) )))
            p.close()
            p.join()
            #counts = Parallel(n_jobs=thread, verbose=0 )(delayed(self.spreadMulti)(x,y,int(times/thread))for _ in range(thread))
            return counts/thread
        else:
            sys.exit("spreadMulti_P wrong") 

    def spreadOnce(self, seedSet_x, seedSet_y, local_state ):
        #local_state = np.random.RandomState()
        
        #local_state.seed()
        '''return # of 0-active nodes'''
        tstate = {} # current state
        fstate = {} # final time
        tTime = dict() # best time
        actTime = [] # all updated time
        for x in seedSet_x:
            tstate[x]=0
            heapq.heappush(actTime, (0, x))
            tTime[x]=0.0
        for y in seedSet_y:
            if y not in seedSet_x:
                tstate[y]=1
                heapq.heappush(actTime, (0, y))
                tTime[y]=0.0
            
        #print(tTime)
        
        while len(actTime)>0:
            current_node_time, current_node = heapq.heappop(actTime)
    
            if current_node not in fstate:
                if current_node_time != tTime[current_node]:
                    sys.exit("current_node_time != tTime[current_node]") 
                    
                fstate[current_node]=current_node_time
                self.spreadLocal(tstate, fstate, actTime, tTime, current_node, current_node_time, local_state)
        count = 0
        for x in tstate:
            if  tstate[x]==0:
                count += 1
        #print(self.vNum-count)
        return count
        
    def spreadLocal(self,tstate, fstate, actTime, tTime, current_node, current_node_time,local_state):
        #print(tTime)
        #print(self.nodes[current_node].neighbor)
        for to_node, para in self.nodes[current_node].neighbor.items():
            if (to_node in fstate) or (not self.isSuccess(self.nodes[to_node], local_state)):
                pass
            else:
                transTime = self.getWeibull(para[0], para[1])
                new_time = current_node_time+ transTime
                if to_node in tstate:
                    
                    if new_time <tTime[to_node]:
                        tTime[to_node]=new_time
                        tstate[to_node]=tstate[current_node]
                        heapq.heappush(actTime, (new_time , to_node))
                        
                    if new_time == tTime[to_node]:
                        if tstate[current_node]==0:
                            tstate[to_node]=0
                            
                if to_node not in tstate:
                   # print(tTime)
                    tTime[to_node]=new_time
                    tstate[to_node]=tstate[current_node]
                    heapq.heappush(actTime, (new_time, to_node))
                    
    def isSuccess(self, to_node, local_state):
        #seed = np.random.seed()
        
        #local_state = np.random.RandomState(seed)
        randnum= local_state.uniform(0, 1)
        if randnum< 1.0/to_node.in_degree:
        #if np.random.uniform(0,1)< 1.0/to_node.in_degree:
            return True
        else:
            return False
        
    def getWeibull(self, alpha, beta):
        time = alpha*math.pow(-math.log(1-random.uniform(0, 1)), beta);
        if time >= 0:
            return math.ceil(time)+1
        else:
            sys.exit("time <0") 
            return None
        
    def genTrains(self, pairsNum, path, simutimes, thread):
        file1 = open('../data/power_list.txt', 'r') 
        seedSizes = []
        while len(seedSizes)<pairsNum: 
            line = file1.readline() 
            if not line: 
                sys.exit("genTrains wrong")
                break          
            seedSizes.append(int(line))
            
        
        with open(path, 'w') as the_file:
            p = multiprocessing.Pool(thread)
            trains = p.starmap(self.getOneTrain, ((seedSizes[i],simutimes) for i in range(pairsNum) ))
            print("pairs generated ")
            p.close()
            p.join()
            for train in trains:
                for x in train.attack:
                    the_file.write(x)
                    the_file.write(" ")
                the_file.write("\n")
                
                for x in train.protect:
                    the_file.write(x)
                    the_file.write(" ")
                the_file.write("\n")
                the_file.write("{}\n".format(train.a_influence))
                the_file.write("{}\n".format(train.p_influence))
                the_file.write("\n")
                
                
                
    def getOneTrain(self, seedSize, simutimes):
        a = self.getRandomSeed(seedSize)
        p,_,_ = self.greedyMP(a,len(a),simutimes)
        a_influence=self.spreadMulti(a, {}, simutimes)
        p_influence=self.spreadMulti(a, p, simutimes)
        return Train(a, p, a_influence, p_influence)
    
    def getRandomSeed(self, seedSize):
        a=set()
        while len(a)<seedSize:
            index = str(math.floor(random.uniform(0, 1)*self.vNum))
            if index not in a:
                a.add(index)
        return a
    
    def greedyMP(self,a,seedSize, simutimes):
        
        c_score = self.spreadMulti_n0(a,{},simutimes)
        
        #print("Initial: {}".format(c_score))
        
        scores = [c_score]
        gains = []
        for node in range(self.vNum):
            gain = self.spreadMulti_n0(a, [str(node)], simutimes) - c_score 
            #print(gain);
            heapq.heappush(gains, (-gain, str(node)))
            
        score_gain, node = heapq.heappop(gains)
        solution = [node]
        
        #score = -score
        c_score = c_score - (-score_gain)
        
        #print("{} + {} + {}".format(node, c_score, -score_gain))
        scores.append(c_score)
    
        # record the number of times the spread is computed
        lookups = [self.vNum]

    
        for _ in range(seedSize - 1):
            node_lookup = 0
            matched = False
    
            while not matched:
                node_lookup += 1
    
                # here we need to compute the marginal gain of adding the current node
                # to the solution, instead of just the gain, i.e. we need to subtract
                # the spread without adding the current node
                _, current_node = heapq.heappop(gains)
                score_gain = self.spreadMulti_n0(a, solution + [current_node], simutimes) - c_score 
    
                # check if the previous top node stayed on the top after pushing
                # the marginal gain to the heap
                heapq.heappush(gains, (-score_gain, current_node))
                matched = gains[0][1] == current_node
            #print(node_lookup)
            # spread stores the cumulative spread
            score_gain, node = heapq.heappop(gains)
            c_score = c_score -  score_gain
            solution.append(node)
            #print("{} + {} + {}".format(node, c_score, -score_gain))
            scores.append(c_score)
            lookups.append(node_lookup)
    

    
        return solution, scores, lookups
            
        
class DiffusionGraph(object):
    '''
    class Node(object):
        def __init__(self,index):
            self.index = index
            self.neighbor = {}
        def print(self):
            for node in self.neighbor:
                print(str(self.index)+" "+str(node)+" "+str(self.neighbor[node]))        
    '''            
    def __init__(self, path_graph, path_distance, vNum):
        self.tranTimes={}
        self.distance = {}
        self.nodes=set()
        self.vNum = vNum
        
        for v in range(self.vNum):
             #node = self.Node(str(v))
             #node.neighbor={}
             neighbor_1={}
             self.tranTimes[str(v)]=neighbor_1
             neighbor_2={}
             self.distance[str(v)]=neighbor_2
             self.nodes.add(str(v))
             
        file1 = open(path_graph, 'r') 
        while True: 
            line = file1.readline() 
            if not line: 
                break          
            strings = line.split()
            node1 = (strings[0])
            node2 = (strings[1])
            time = float(strings[2])
            
            if node1 in self.tranTimes:
                self.tranTimes[node1][node2]=time
            else:
                sys.exit("non existing node") 
                
            if node2 not in self.nodes:
                sys.exit("non existing node") 
                
       
        
        file1 = open(path_distance, 'r') 
        while True: 
            line = file1.readline() 
            if not line: 
                break          
            strings = line.split()
            node1 = (strings[0])
            node2 = (strings[1])
            time = float(strings[2])
            
            if node1 in self.distance:
                self.distance[node1][node2]=time
            else:
                sys.exit("non existing node") 
                
            if node2 not in self.nodes:
                sys.exit("non existing node") 
      
        
    def print(self):
        #for node in self.nodes:
            #print(self.tranTimes[node])
        for node in self.nodes:
            print(self.distance[node])

    def spread(self, seedSet_x, seedSet_y, getCover=False):
        '''return # of 0-active nodes'''
        tstate = {} # current state
        fstate = {} # final time
        tTime = dict() # best time
        actTime = [] # all updated time
        for x in seedSet_x:
            tstate[x]=0
            heapq.heappush(actTime, (0, x))
            tTime[x]=0.0
        for y in seedSet_y:
            if y not in seedSet_x:
                try:
                    tstate[y]=1
                    heapq.heappush(actTime, (0, y))
                except: 
                    print(y)
                    print(seedSet_y)
                    input("Press Enter to continue...")
                
                tTime[y]=0.0
            
        #print(tTime)
        
        while len(actTime)>0:
            current_node_time, current_node = heapq.heappop(actTime)
            
            
                
            if current_node not in fstate:
                if current_node_time != tTime[current_node]:
                    sys.exit("current_node_time != tTime[current_node]") 
                fstate[current_node]=current_node_time
                self.spreadLocal(tstate, fstate, actTime, tTime, current_node, current_node_time)
        count = 0
        cover={}
        for x in tstate:
            if  tstate[x]==0:
                count += 1
                cover[x]=tTime[x]
        #print(self.vNum-count)
        if getCover:
            return count, cover
        else:
            return count
        
    def spreadLocal(self,tstate, fstate, actTime, tTime, current_node, current_node_time):
        #print(tTime)
        #print(self.nodes[current_node].neighbor)
        for to_node in self.tranTimes[current_node]:
            tranTime=self.tranTimes[current_node][to_node]
            if to_node in fstate:
                pass
            else:
                new_time = current_node_time+ tranTime
                if to_node in tstate:
                    
                    if new_time <tTime[to_node]:
                        tTime[to_node]=new_time
                        tstate[to_node]=tstate[current_node]
                        heapq.heappush(actTime, (new_time , to_node))
                        
                    if new_time == tTime[to_node]:
                        if tstate[current_node]==0:
                            tstate[to_node]=0
                            
                if to_node not in tstate:
                   # print(tTime)
                    tTime[to_node]=new_time
                    tstate[to_node]=tstate[current_node]
                    heapq.heappush(actTime, (new_time, to_node))
                    
    def getDistance(self, oneset, node):
        distance = sys.maxsize
        for x in oneset:
            if node in self.distance[x]:
                if self.distance[x][node]<distance:
                    distance=self.distance[x][node]
        return distance
        
        
    #class Edge(object):
     #   pass
    
    
    
class InputInstance(object):
    
    def __init__(self, socialGraphPath, featurePath, featureNum, vNum, effectAreaHotNum, balance_para, loss_type, 
                 featureRandom = False, maxFeatureNum = 500, thread = 1, LAI_method = None, indexes=None):
        #self.graphs=[];
        self.featureNum = featureNum
        self.vNum = vNum
        self.socialGraph = SocialGraph(socialGraphPath, vNum);
        #self.socialGraph.print()
        #print(self.socialGraph.nodes)
        self.effectAreaHotNum = effectAreaHotNum;
        self.balance_para = balance_para
        self.thread=thread
        self.LAI_method = LAI_method 
        if loss_type != None:
            self.loss_type=loss_type.name
            if loss_type.name == "hamming": 
                self.hammingWeight=loss_type.weight
        self.hammingWeight = None
        self.featureRandom = featureRandom
        self.maxFeatureNum = maxFeatureNum
        
        # self.readFeatures(path, featureNum)
        # read social graph
       
       #read features
        self.diffusionGraphs = [];
        if indexes != None:
            lineNums=indexes
            self.featureIndexes=lineNums
            #print("lineNums: {}".format(lineNums))
            for i in lineNums:
                path_graph="{}{}_graph.txt".format(featurePath, i)
                path_distance="{}{}_distance.txt".format(featurePath, i)
                diffusionGraph = DiffusionGraph(path_graph,path_distance, vNum)
                self.diffusionGraphs.append(diffusionGraph)
        else:
            if self.featureRandom:
                lineNums=(np.random.permutation(maxFeatureNum))[0:featureNum]
                self.featureIndexes=lineNums
                #print("lineNums: {}".format(lineNums))
                for i in lineNums:
                    path_graph="{}{}_graph.txt".format(featurePath, i)
                    path_distance="{}{}_distance.txt".format(featurePath, i)
                    diffusionGraph = DiffusionGraph(path_graph,path_distance, vNum)
                    self.diffusionGraphs.append(diffusionGraph)
            else:
                for i in range(featureNum):
                    path_graph="{}{}_graph.txt".format(featurePath, i)
                    path_distance="{}{}_distance.txt".format(featurePath, i)
                    diffusionGraph = DiffusionGraph(path_graph,path_distance, vNum)
                    self.diffusionGraphs.append(diffusionGraph)
                    #diffusionGraph.print()
        
    
        
    def computeFeature(self, x, y):
        feature = [];
        for graph in self.diffusionGraphs:
            feature.append(self.computeScoreOneGraph(x, y, graph))
        return np.array(feature) 
    
    def computeScore(self, x, y, w):
        feature = self.computeFeature(x, y)
        return w.dot(feature)
    
    def computeScoreOneGraph(self, x, y, graph):    
        '''compute f^g(M,P)'''
        return self.vNum-graph.spread(x, y)
        
    
    
    def inference(self, x, w):
        #print("inference") 
        start_time = time.time()
        
        c_score = self.computeScore(x, [ ], w)
        
        #print("Initial: {}".format(c_score))
        
        scores = [c_score]
        gains = []
        for node in range(self.vNum):
            gain = self.computeScore(x, [str(node)], w) - c_score 
            #print(gain);
            heapq.heappush(gains, (-gain, str(node)))
            
        score_gain, node = heapq.heappop(gains)
        solution = [node]
        
        #score = -score
        c_score = c_score - (score_gain)
        #print("{} {}".format(node, -score_gain)) 
        #print("{} + {} + {}".format(node, c_score, -score_gain))
        scores.append(c_score)
    
        # record the number of times the spread is computed
        lookups = [self.vNum]
        elapsed = [round(time.time() - start_time, 3)]
    
        for _ in range(len(x) - 1):
            node_lookup = 0
            matched = False
    
            while not matched:
                node_lookup += 1
    
                # here we need to compute the marginal gain of adding the current node
                # to the solution, instead of just the gain, i.e. we need to subtract
                # the spread without adding the current node
                _, current_node = heapq.heappop(gains)
                score_gain = self.computeScore(x, solution + [current_node], w) - c_score 
    
                # check if the previous top node stayed on the top after pushing
                # the marginal gain to the heap
                heapq.heappush(gains, (-score_gain, current_node))
                matched = gains[0][1] == current_node
            #print(node_lookup)
            # spread stores the cumulative spread
            score_gain, node = heapq.heappop(gains)
            c_score = c_score -  score_gain
            solution.append(node)
            #print("{} {}".format(node, -score_gain)) 
            #print("{} + {} + {}".format(node, c_score, -score_gain))
            scores.append(c_score)
            lookups.append(node_lookup)
    
            elapse = round(time.time() - start_time, 3)
            elapsed.append(elapse)
    
        return solution, scores, elapsed, lookups
    
    def inferenceRandom(self,x,w):
        solution = []
        nodeSet = []
        for i in range(self.vNum):
            nodeSet.append(str(i))
        while len(solution) < len(x):
            node = random.choice(nodeSet)
            if node not in x:
                solution.append(node)
                
        #print(self.computeScore(x, solution , w) )
        return solution
        
    def loss(self, y, y_hat):
        if self.loss_type == None:
            sys.exit("loss method not speficied.") 
        if self.loss_type == "area":
            return self.similarity(y, y)-self.similarity(y, y_hat)
        if self.loss_type == "hamming":
            if y == y_hat:
                return 0
            else:
                if self.hammingWeight == None:
                    sys.exit("hammingWeight == None") 
                return self.hammingWeight

    
    def similarity(self, x, y):
        set1 = self.socialGraph.getNeighborsByHot(x, 1)
        set2 = self.socialGraph.getNeighborsByHot(y, 1)
        
        return len(set1.intersection(set2))
    
    
    
    def loss_augmented_inference(self, x, y ,w):
        if self.loss_type == None:
            sys.exit("loss_augmented_inference method not speficied.") 
            
        if self.loss_type == "area":
            if self.LAI_method == "greedy":
                return self.loss_augmented_inference_area_greedy(x, y, w)
            if self.LAI_method == "lazy":
                return self.loss_augmented_inference_area_greedy_lazy(x, y, w)
            if self.LAI_method == "fastLazy":
                return self.loss_augmented_inference_area_greedy_lazy_fast(x, y, w)
            if self.LAI_method == "fastGreedy":
                return self.loss_augmented_inference_area_greedy_fast(x, y, w)
            
        if self.loss_type == "hamming":
            return self.loss_augmented_inference_hamming(x, y, w)

            
        
            
    def loss_augmented_inference_objective(self, x, y, y_pre, w):
        inference = self.computeScore(x,y_pre,w)
        loss = self.loss(y,y_pre)
        #print("{} + {}".format(inference, loss))
        return inference+self.balance_para*loss
    
    
    
    def loss_augmented_inference_area_greedy(self, x, y ,w):
        #print("loss_augmented_inference_greedy")
        solution = set()
        for i in range(len(x)):
            c_value = 0
            c_index = None
            for v in range(self.vNum):
                value = self.loss_augmented_inference_objective(x, y, solution.union({str(v)}), w)
                if value>=c_value:
                    c_index=str(v)
                    c_value=value
 
            #print("{} {}".format(c_index,c_value-self.loss_augmented_inference_objective(x, y, solution, w)))
            solution.add(c_index)
            #print(solution)
        #print(self.loss_augmented_inference_objective(x, y, solution, w))
        return solution
    
    def loss_augmented_inference_area_greedy_fast(self, x, y ,w):
        #print("loss_augmented_inference_greedy_fast")
        solution = set()
        c_cover = []
        temp = []
        for graph in self.diffusionGraphs:
            tempp, c_coverOneGraph=graph.spread(x,{}, getCover=True)
            c_cover.append(c_coverOneGraph)
            temp.append(tempp)
            #print(c_coverOneGraph)
            

        for i in range(len(x)):
            c_value = 0
            c_index = None
            t_cover = {};
            for v in range(self.vNum):
                value, node_cover = self.loss_augmented_inference_fast_scoreGain(x, y, solution, {str(v)}, w, c_cover)
                if value>=c_value:
                    c_index=str(v)
                    c_value=value
                    t_cover = node_cover
 
            #print("{} {}".format(c_index,c_value))
            solution.add(c_index)
            c_cover = t_cover
            #print(solution)
        #print(self.loss_augmented_inference_objective(x, y, solution, w))
        return solution
    
    def loss_augmented_inference_hamming(self, x, y ,w):
        y1, scores,_,_ = self.inference(x, w)
        score2 = self.computeScore(x, y, w)
        if scores[-1]+self.loss(y, y1) > score2:
            return y1
        else:
            return y
        
    def loss_augmented_inference_area_greedy_lazy(self, x, y ,w):
        #print("loss_augmented_inference_area_fakeLazeGreedy")
        solution = set()
        gains = []
        for node in range(self.vNum):
            gain = self.loss_augmented_inference_objective(x, y, solution.union({str(node)}), w)
            #print(gain);
            heapq.heappush(gains, (-gain, str(node)))
            
        score_gain, node = heapq.heappop(gains)
        solution.add(node)
        c_score = -score_gain
        #print("{} {}".format(node, -score_gain)) 
    
        for _ in range(len(x) - 1):

            matched = False
    
            while not matched:
                _, current_node = heapq.heappop(gains)
                score_gain = self.loss_augmented_inference_objective(x, y, solution.union({current_node}), w) - c_score 
                heapq.heappush(gains, (-score_gain, current_node))
                matched = gains[0][1] == current_node

            score_gain, node = heapq.heappop(gains)
            c_score = c_score -  score_gain
            solution.add(node)
            #print("{} {}".format(node, -score_gain)) 
        #print(self.loss_augmented_inference_objective(x, y, solution, w))
        return solution
    
    def loss_augmented_inference_area_greedy_lazy_fast(self, x, y ,w):
        #print("loss_augmented_inference_area_fast")
        solution = set()
        gains = []
        c_cover = []
        temp = []
        for graph in self.diffusionGraphs:
            tempp, c_coverOneGraph=graph.spread(x,{}, getCover=True)
            c_cover.append(c_coverOneGraph)
            temp.append(tempp)
            #print(c_coverOneGraph)
            
        for node in range(self.vNum):
            gain, node_cover = self.loss_augmented_inference_fast_scoreGain(x, y, solution, {str(node)}, w, c_cover)
            #input("Press Enter to continue...")
            #print(-gain)  
            heapq.heappush(gains, (-gain, str(node), node_cover))
        # 
        score_gain, node, node_cover = heapq.heappop(gains)
        solution.add(node)
        c_score = -score_gain
        #print("{} {}".format(node, -score_gain)) 

        c_cover=node_cover
        
        for _ in range(len(x) - 1):
            matched = False
            while not matched:
                _, current_node, _ = heapq.heappop(gains)
                score_gain, new_cover = self.loss_augmented_inference_fast_scoreGain(x, y, solution, {str(current_node)}, w, c_cover)
                #if score_gain <0:
                #print("score_gain {}".format(score_gain))
                heapq.heappush(gains, (-score_gain, current_node, new_cover))
                matched = gains[0][1] == current_node

            score_gain, node, c_cover = heapq.heappop(gains)
            c_score = c_score -  score_gain
            solution.add(node)
            #print("{} {}".format(node, -score_gain)) 
        #print(self.loss_augmented_inference_objective(x, y, solution, w))
        return solution
    
    def loss_augmented_inference_fast_scoreGain(self, x, y, current, newset, w, c_cover):
        inferenceGain, new_cover = self.computeScoreGain(x, y, current, newset, w, c_cover)
        lossGain = self.loss(y,current.union(newset))-self.loss(y,current)
        #print("{} + {}".format(inference, loss))
        return inferenceGain+self.balance_para*lossGain, new_cover
    
    def computeScoreGain(self, x, y,current, newset, w, c_cover):
        scoreGain = []
        new_cover = []
        #print(c_cover) 
        for graph, c_coverOneGraph in zip(self.diffusionGraphs, c_cover):
            gain, newcoverOneGraph=self.computeScoreGainOneGraph(x, y,current, newset, c_coverOneGraph, graph)
            #print("gain {}".format(gain))
            scoreGain.append(gain)
            new_cover.append(newcoverOneGraph)
        #print("scoreGain {}".format(scoreGain))
        return w.dot(np.array(scoreGain)), new_cover
    
    def computeScoreGainOneGraph(self, x, y, current, newset, c_coverOneGraph, graph):
        dnames=[]
        newcoverOneGraph=c_coverOneGraph.copy()
        for node in c_coverOneGraph:
            #print(c_coverOneGraph[node])
            if graph.getDistance(newset,node)<c_coverOneGraph[node]:
                #input("Press Enter to continue...")
                dnames.append(node)
        for name in dnames:
            del newcoverOneGraph[name]
            
        #if len(c_coverOneGraph) != len(newcoverOneGraph):
         #          print("!!!! {} {}".format(len(c_coverOneGraph),len(newcoverOneGraph)))
        return len(c_coverOneGraph)-len(newcoverOneGraph), newcoverOneGraph
    
    def testInfluence_0(self, x, y, times, thread):
        if thread>1:
            return self.socialGraph.spreadMulti_P(x,y,times, thread)
        else:
            return self.socialGraph.spreadMulti(x,y,times)
        
    def testInfluence_0_block(self, X, times, Y = None):
        result = []
        if Y == None:
            for x in X:
                result.append(self.socialGraph.spreadMulti(x,{},times))
        else:
            for x, y in zip(X,Y):
                result.append(self.socialGraph.spreadMulti(x,y,times))
        return result

            

    
class StratLearn(StructuredModel):
    """Interface definition for Structured Learners.

    This class defines what is necessary to use the structured svm.
    You have to implement at least joint_feature and inference.
    """
    def __repr__(self):
        return ("%s, size_joint_feature: %d"
                % (type(self).__name__, self.size_joint_feature))

    def __init__(self):
        """Initialize the model.
        Needs to set self.size_joint_feature, the dimensionality of the joint
        features for an instance with labeling (x, y).
        """
        self.size_joint_feature = None

    def _check_size_w(self, w):
        if w.shape != (self.size_joint_feature,):
            raise ValueError("Got w of wrong shape. Expected %s, got %s" %
                             (self.size_joint_feature, w.shape))

    def initialize(self, X, Y, instance):
        # set any data-specific parameters in the model
        #self.featureNum = instance.featureNum
        self.size_joint_feature= instance.featureNum
        self.instance = instance
        self.inference_calls = 0
        #if self.n_features is None:
        #    self.n_features = n_features
        #elif self.n_features != n_features:
        #    raise ValueError("Expected %d features, got %d"
        #                     % (self.n_features, n_features))

        #n_labels = Y.shape[1]
        #if self.n_labels is None:
        #    self.n_labels = n_labels
        #elif self.n_labels != n_labels:
        #    raise ValueError("Expected %d labels, got %d"
          #                   % (self.n_labels, n_labels))

        #self._set_size_joint_feature()
        #self._set_class_weight()
        pass
    """
    def joint_feature(self, x, y):
        raise NotImplementedError()
    """ 
    def joint_feature(self, x, y):
        return self.instance.computeFeature(x,y)
        '''
        feature = np.zeros(self.featureNum)
        index = 0
        for graph in self.instance.graphs:
            distance_matrix = np.zeros( (2, 3) )
            for v in range(self.instance.nNUm):
                x_min=sys.maxsize
                for u in x:
                    if distance_matrix[v][u]<x_min:
                        x_min=distance_matrix[v][u]
                y_min=sys.maxsize
                for u in y:
                    if distance_matrix[v][u]<y_min:
                        y_min=distance_matrix[v][u]
                if y_min<x_min:
                    feature[index] += 1
            index += 1
        return feature
        '''

    def batch_joint_feature(self, X, Y, Y_true=None):
        #print("batch_joint_feature running")
        joint_feature_ = np.zeros(self.size_joint_feature)
        if getattr(self, 'rescale_C', False):
            for x, y, y_true in zip(X, Y, Y_true):
                joint_feature_ += self.joint_feature(x, y, y_true)
        else:
            for x, y in zip(X, Y):
                joint_feature_ += self.joint_feature(x, y)
        #print("batch_joint_feature done")
        return joint_feature_

    def _loss_augmented_djoint_feature(self, x, y, y_hat, w):
        # debugging only!
        x_loss_augmented = self.loss_augment(x, y, w)
        return (self.joint_feature(x_loss_augmented, y)
                - self.joint_feature(x_loss_augmented, y_hat))
    
    def inference_block(self, X, w,relaxed=None, constraints=None):
        Y = []
        for x in X:
            Y.append(self.inference(x, w, relaxed, constraints))
        return Y

    def inference(self, x, w, relaxed=None, constraints=None):
        self.inference_calls += 1
        solution,_,_,_ = self.instance.inference(x,w)
        return solution
        #raise NotImplementedError()

    def batch_inference(self, X, w, relaxed=None, constraints=None):
        # default implementation of batch inference
        if constraints:
            return [self.inference(x, w, relaxed=relaxed, constraints=c)
                    for x, c in zip(X, constraints)]
        return [self.inference(x, w, relaxed=relaxed)
                for x in X]

    def loss(self, y, y_hat):
        '''
        # hamming loss:
        if isinstance(y_hat, tuple):
            return self.continuous_loss(y, y_hat[0])
        if hasattr(self, 'class_weight'):
            return np.sum(self.class_weight[y] * (y != y_hat))
        return np.sum(y != y_hat)
        '''
        return self.instance.loss(y,y_hat)
        

    def batch_loss(self, Y, Y_hat):
        # default implementation of batch loss
        return [self.loss(y, y_hat) for y, y_hat in zip(Y, Y_hat)]

    def max_loss(self, y):
        # maximum possible los on y for macro averages
        sys.exit("max_loss not implemented") 
        if hasattr(self, 'class_weight'):            return np.sum(self.class_weight[y])
        return y.size

    def continuous_loss(self, y, y_hat):
        # continuous version of the loss
        # y is the result of linear programming
        sys.exit("continuous_loss not implemented") 
        if y.ndim == 2:
            raise ValueError("FIXME!")
        gx = np.indices(y.shape)

        # all entries minus correct ones
        result = 1 - y_hat[gx, y]
        if hasattr(self, 'class_weight'):
            return np.sum(self.class_weight[y] * result)
        return np.sum(result)

    def loss_augmented_inference(self, x, y, w, relaxed=None):
        
        #print("FALLBACK no loss augmented inference found")
        #return self.inference(x, w)
        #print("loss_augmented_inference RUNNING")
        self.inference_calls += 1
        y_pre = self.instance.loss_augmented_inference(x,y,w)
        #print("loss_augmented_inference DONE")
        return y_pre
    
    def loss_augmented_inference_block(self, X, Y, w, relaxed=None):
        
        #print("FALLBACK no loss augmented inference found")
        #return self.inference(x, w)
        #print("loss_augmented_inference RUNNING")
        self.inference_calls += len(X)
        result =[]
        for x, y in zip(X,Y):
            result.append(self.instance.loss_augmented_inference(x,y,w))
        return result
    

        

    def batch_loss_augmented_inference(self, X, Y, w, relaxed=None):
        #sys.exit("batch_loss_augmented_inference not implemented") 
        # default implementation of batch loss augmented inference
        return [self.loss_augmented_inference(x, y, w, relaxed=relaxed)
                for x, y in zip(X, Y)]

    def _set_class_weight(self):
        sys.exit("_set_class_weight not implemented") 
        if not hasattr(self, 'size_joint_feature'):
            # we are not initialized yet
            return

        if hasattr(self, 'n_labels'):
            n_things = self.n_labels
        else:
            n_things = self.n_states

        if self.class_weight is not None:

            if len(self.class_weight) != n_things:
                raise ValueError("class_weight must have length n_states or"
                                 " be None")
            self.class_weight = np.array(self.class_weight)
            self.uniform_class_weight = False
        else:
            self.class_weight = np.ones(n_things)
            self.uniform_class_weight = True
            
