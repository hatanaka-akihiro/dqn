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
		"""
		指定枚数カードを配る
		2次元配列で返す
		[[1枚目の記号, 1枚目の数字], [2枚目の記号, 2枚目の数字],...]
		"""
		cards = []
		for i in range(num):
			suit, num = self.getCard()
			cards.append([suit, num])
		return self.__sortCards(cards)

	def __sortCards(self, cards):
		"""
		カードを数字の昇順に並べる
		"""
		cards.sort(key = lambda x:x[1])
		return cards

	def getCard(self):
		"""
		カードを1枚配る
		記号 (0-3) と数字(0-12) を返す
		"""
		while(True):
			card = np.random.randint(0, NUM_OF_CARDS)
			if self.CARDS[card] == 0:
				self.CARDS[card] += 1
				# print('getCard: %02d' % card)
				return int(card / 13), card % 13
	
	def reset(self):
		self.CARDS = self.__resetCards(NUM_OF_CARDS)

	def draw(self, cards, changes):
		"""
		指定のカードを捨てて、新しいカードを配る
		cards : 手持ちのカード
		changes : 捨てるカードを2進数で指定。iビット目が1の場合、i枚目を捨てる。0b11100 の場合、3-5枚目を捨てる。
		return : 捨てたカードに、新しいカードを加えて返す。getCards() と同じ。
		"""
		for i in range(len(cards)):
			check = 0b1 << i
			if check & changes == check:
				suit, num = self.getCard()
				cards[i] = [suit, num]
		return self.__sortCards(cards)

	def getScore(self, cards):
		"""
		カードの点数を計算して返す
		"""
		pears = self.__resetCards(13)
		for card in cards:
			num = card[1]
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
		"""
		カードを outfile に出力する
		"""
		for card in cards:
			suit = card[0]
			num = card[1]
			outfile.write('%c%02d ' % (Deck.MARKS[suit], num + 1))
		outfile.write('\n')
		
class Poker(gym.core.Env):
	metadata = {'render.modes': ['human']}
	def __init__(self):
		self.action_space = gym.spaces.Discrete(2 ** 5)
		self.observation_space = gym.spaces.MultiDiscrete([[0,3],[0,12], [0,3],[0,12], [0,3],[0,12], [0,3],[0,12], [0,3],[0,12]])
		self._deck = Deck()

	def _step(self, action):
		"""
		各stepごとに呼ばれる
		actionを受け取り、次のstateとreward、episodeが終了したかどうかを返すように実装
		"""
		
		self.render()

		# カード交換前の報酬を計算
		prev_reward = self._getScore()

		# カード交換
		print("changes: %s" % format(action, 'b')[::-1])
		self._cards = self._deck.draw(self._cards, action)
		self.render()
		
		# カード交換後の報酬を計算
		reward = self._getScore()

		# 交換によって報酬が下がったら、マイナスにする		
		if prev_reward > reward:
			reward = reward - prev_reward

		print("reward: %d" % reward)

		# 常に終了
		done = True

		return self.__convertCards(), reward, done, {}

	# 各episodeの開始時に呼ばれ、初期stateを返すように実装
	def _reset(self):
		self._deck.reset()
		self._cards = self._deck.getCards(5)
		return self.__convertCards()
	
	def _getScore(self):
		return self._deck.getScore(self._cards)

	def _render(self, mode='human', close=False):
		if close:
			return
		# outfile = StringIO() if mode == 'ansi' else sys.stdout
		outfile = sys.stdout
		self._deck.toString(outfile, self._cards)
		if mode != 'human':
			return outfile
	
	def __convertCards(self):
		"""
		手持ちのカードを表す self._cards (2次元配列) を1次元配列に変換して返す
		"""
		return [flatten for inner in self._cards for flatten in inner]

#outfile = sys.stdout
#deck = Deck()
#cards = deck.getCards(5)
#deck.toString(outfile, cards)
#cards = deck.draw(cards, [0, 2, 4])
#deck.toString(outfile, cards)
#print("score: %d" % deck.getScore(cards))

#poker = Poker()
#poker.reset()
#poker.step(10)