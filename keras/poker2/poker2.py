import sys
import numpy as np
import gym
import gym.spaces

NUM_OF_CARDS = 13 * 4
NUM_OF_COMBINATION = 2598960

class Deck():
	MARKS = ['S', 'H', 'D', 'C']

	def __init__(self):
		self.reset()
		#print(self.CARDS)
	
	def getCards(self, num):
		cards = []
		for i in range(num):
			suit, num = self.getCard()
			cards.append([suit, num])
		# 数字順に並べる
		cards.sort(key = lambda x:x[1])
		return cards

	def getCard(self):
		while(True):
			card = np.random.randint(0, NUM_OF_CARDS)
			if self.CARDS[card] == 0:
				self.CARDS[card] += 1
				# print('getCard: %02d' % card)
				return int(card / 13), card % 13
	
	def reset(self):
		self.CARDS = self.__resetCards(NUM_OF_CARDS)

	def draw(self, cards, changes):
		for i in changes:
			suit, num = self.getCard()
			cards[i] = [suit, num]
		# 数字順に並べる
		cards.sort(key = lambda x:x[1])
		return cards

	def getScore(self, cards):
		pears = self.__resetCards(13)
		for i in self.__toNums(cards):
			num = cards[i][1]
			pears[num] += 1
		score = 0
		two = 0
		three = 0
		four = 0
		for i in pears:
			if i == 2:
				two += 1
			elif i == 3:
				three += 1
			elif i == 4:
				four += 1
		if two == 1:
			if three == 1:
				# full house
				return NUM_OF_COMBINATION / 5108
			else:
				return NUM_OF_COMBINATION / 1098240
		if two == 2:
			return NUM_OF_COMBINATION / 123552
		if three == 1:
			return NUM_OF_COMBINATION / 54912
		if four == 1:
			return NUM_OF_COMBINATION / 624 
		return 0
		
	def __resetCards(self, length):
		return np.zeros(length, dtype=np.int)
	
	def toString(self, outfile, cards):
		for i in range(len(cards)):
			suit = cards[i][0]
			num = cards[i][1]
			outfile.write('%c%02d ' % (Deck.MARKS[suit], num + 1))
		outfile.write('\n')

	def __toNums(self, cards):
		nums = []
		for i in range(len(cards)):
			if cards[i] != 0:
				nums.append(i)
		#print(nums)
		return nums
		
class Poker(gym.core.Env):
	metadata = {'render.modes': ['human']}
	def __init__(self):
		self.action_space = gym.spaces.Discrete(2 ** 5)
		self.observation_space = gym.spaces.MultiDiscrete([[0,3],[0,12], [0,3],[0,12], [0,3],[0,12], [0,3],[0,12], [0,3],[0,12]])
		self._deck = Deck()

	# 各stepごとに呼ばれる
	# actionを受け取り、次のstateとreward、episodeが終了したかどうかを返すように実装
	def _step(self, action):
		self.render()
		changes = self.__bitsToNums(action, 5)
		print("changes: %s" % changes)
		self._cards = self._deck.draw(self._cards, changes)
		reward = self._deck.getScore(self._cards)
		self.render()
		print("reward: %d" % reward)
		done = True
		return self.getCards(), reward, done, {}

	# 各episodeの開始時に呼ばれ、初期stateを返すように実装
	def _reset(self):
		self._deck.reset()
		self._cards = self._deck.getCards(5)
		return self.getCards()

	def _render(self, mode='human', close=False):
		if close:
			return
		# outfile = StringIO() if mode == 'ansi' else sys.stdout
		outfile = sys.stdout
		self._deck.toString(outfile, self._cards)
		if mode != 'human':
			return outfile
	
	def getCards(self):
		return [flatten for inner in self._cards for flatten in inner]

	def __bitsToNums(self, bits, length):
		nums = []
		for i in range(length):
			bit = 0b1 << i
			if bits & bit == bit:
				nums.append(i)
		return nums

#outfile = sys.stdout
#deck = Deck()
#cards = deck.getCards(5)
#deck.toString(outfile, cards)
#cards = deck.draw(cards, [0, 2, 4])
#deck.toString(outfile, cards)
#print("score: %d" % deck.getScore(cards))

poker = Poker()
poker.reset()
poker.step(10)