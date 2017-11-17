import numpy as np
import gym
import gym.spaces

NUM_OF_CARDS = 13 * 4

class Deck():
	MARKS = ['S', 'H', 'D', 'C']

	def __init__(self):
		self.reset()
		print(self.CARDS)
	
	def getCards(self, num):
		cards = [];
		for i in range(num):
			cards.append(self.getCard())
		return self.__arrayToNum(cards)

	def getCard(self):
		while(True):
			card = np.random.randint(0, NUM_OF_CARDS)
			bit = 0b1 << card
			if self.CARDS & bit == 0b0:
				self.CARDS = self.CARDS | bit
				print('getCard: %02d' % card)
				return card
	
	def reset(self):
		self.CARDS = 0b0

	def draw(self, cards, changes):
		cards = self.__numToArray(cards)
		changes = self.__numToArray(changes, 5)	
		for i in changes:
			cards[i] = getCard()
		return self.__arrayToNum(cards)

	def getScore(self, cards):
		cards = self.__numToArray(cards)
		pears = np.zeros(13)
		for i in cards:
			num = i % 13
			pears[num] += 1
		score = 0
		for i in pears:
			if i > 1:
				score += i * 10
		return score
	
	def __numToArray(self, num, length = NUM_OF_CARDS):
		cards = []
		for i in range(length):
			bit = 0b1 << i
			if bit == num & bit:
				cards.append(i)
		cards.sort()
		print(cards)
		return cards

	def __arrayToNum(self, array, length = NUM_OF_CARDS):
		num = 0b0
		for i in array:
			num = num | 0b1 << i
		return num

	def printCards(self, cards):
		cards = self.__numToArray(cards)
		for i in cards:
			mark = int(i / 13)
			num = i%13 + 1
			print('%c%02d' % (Deck.MARKS[mark], num), end=' ')
		print()

# 直線上を動く点の速度を操作し、目標(原点)に移動させることを目標とする環境
class Poker(gym.core.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(5) 
        self.observation_space = gym.spaces.Discrete(NUM_OF_CARDS)
        self._deck = Deck()

    # 各stepごとに呼ばれる
    # actionを受け取り、次のstateとreward、episodeが終了したかどうかを返すように実装
    def _step(self, action):
        self._cards = self._deck.draw(self._cards, action) 
        reward = self._deck.getScore(self._cards)
        done = true
        return self._cards, reward, done, {}

    # 各episodeの開始時に呼ばれ、初期stateを返すように実装
    def _reset(self):
        # 初期stateは、位置はランダム、速度ゼロ
        self._deck.reset()
        self._cards = self._deck.getCards(5)
        return self._cards

deck = Deck()
cards = deck.getCards(5)
deck.printCards(cards)
print("score: %d" % deck.getScore(cards))