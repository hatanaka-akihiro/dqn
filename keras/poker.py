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
		cards = self.__resetCards(NUM_OF_CARDS)
		for i in range(num):
			card = self.getCard()
			cards[card] += 1
		return cards

	def getCard(self):
		while(True):
			card = np.random.randint(0, NUM_OF_CARDS)
			if self.CARDS[card] == 0:
				self.CARDS[card] += 1
				print('getCard: %02d' % card)
				return card
	
	def reset(self):
		self.CARDS = self.__resetCards(NUM_OF_CARDS)

	def draw(self, cards, changes):
		cards = self.__toNums(cards)
		changes = self.__toNums(changes)	
		for i in changes:
			cards[i] = self.getCard()
		return self.__fromNum(cards)

	def getScore(self, cards):
		pears = self.__resetCards(13)
		for i in self.__toNums(cards):
			num = int(i % 13)
			pears[num] += 1
		score = 0
		for i in pears:
			if i > 1:
				score += i * 10
		return score
		
	def __resetCards(self, length):
		return np.zeros(length, dtype=np.int)
	
	def __toNums(self, cards):
		nums = []
		for i in range(len(cards)):
			if cards[i] != 0:
				nums.append(i)
		print(nums)
		return nums

	def __fromNum(self, nums, length = NUM_OF_CARDS):
		cards = self.__resetCards(length)
		for i in nums:
			cards[i] += 1
		return cards

	def printCards(self, cards):
		for i in range(len(cards)):
			if cards[i] == 1:
				mark = int(i / 13)
				num = i%13 + 1
				print('%c%02d' % (Deck.MARKS[mark], num), end=' ')
		print()

# 直線上を動く点の速度を操作し、目標(原点)に移動させることを目標とする環境
class Poker(gym.core.Env):
    def __init__(self):
        self.action_space = gym.spaces.MultiBinary(5) 
        self.observation_space = gym.spaces.MultiBinary(NUM_OF_CARDS)
        self._deck = Deck()

    # 各stepごとに呼ばれる
    # actionを受け取り、次のstateとreward、episodeが終了したかどうかを返すように実装
    def _step(self, action):
        self._cards = self._deck.draw(self._cards, action) 
        reward = self._deck.getScore(self._cards)
        done = True
        return self._cards, reward, done, {}

    # 各episodeの開始時に呼ばれ、初期stateを返すように実装
    def _reset(self):
        self._deck.reset()
        self._cards = self._deck.getCards(5)
        return self._cards

# deck = Deck()
# cards = deck.getCards(5)
# deck.printCards(cards)
# cards = deck.draw(cards, [1, 0, 1, 0, 1])
# deck.printCards(cards)
# print("score: %d" % deck.getScore(cards))

poker = Poker()
poker.reset()
print(poker.step([0, 1, 0, 1, 0]))