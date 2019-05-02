# ===============================================================
#											
#				CSCI3202 Sp 2019 Practicum:
#				Christian Simons, Ali Noor
#
# 		Python 9x9 Sudoku Solver using CSP techniques:
#	
#			1. Forward Checking of Domain Space
#			2. Create Back Track tree level based on
#			   available domain choices.
#			3. Repeat until the Goal State is reached.
#
# ===============================================================

'''
sudoku.sdk format:


# Board Format:

		
		c0 c1 c2   c3 c4 c5   c6 c7 c8
		
		   N0		  N1		 N2
r0		0  0  0    0  6  0    0  0  0
r1	 	9  0  0    0  0  0    0  0  0
r2		0  0  0    2  8  3    0  0  5

		   N3		  N4		 N5
r3		0  0  5    0  0  0    9  0  8
r4		0  0  0    0  0  0    6  1  0
r5		0  1  0    3  0  0    0  0  0

		   N6		  N7		 N8
r6		0  2  9    1  0  0    5  0  0
r7		0  0  3    5  0  0    4  0  0
r8		0  6  0    8  2  0    0  0  7

Pattern: r[x], c[y]: r[x] = y, c[y] = x  => Row's element number is mapped via the column number.

Infrastructure for the solver algorithm:
1. Forward Check Domains for rows, columns, and nCells, then pick a number for that spot
2. Create a new level in the Back Tracking Tree with the new game states.
3. Check leaves if is goal state, if it is terminate. If not, go back to step 1, repeat.

# r Row (R[0]) (Left to Right)	=> R0 Domain = {1, 2, 3, 4, 5, 7, 8, 9}
  0 0 0 0 6 0 0 0 0
  
# c Column (C[0]) (Top to Bottom)	=> C0 Domain = {1, 2, 3, 4, 5, 6, 7, 8}
  0 9 0 0 0 0 0 0 0
  
   
# N Cell (N[0])  => N0 Domain = {1, 2, 3, 4, 5, 6, 7, 8}
  0 0 0
  9 0 0
  0 0 0
  
  Cell N0 contains R[0->2], C[0->2]
  
  
  Full Analysis of board structure:
   
  N cell Orientation:
  N0 N1 N2
  N3 N4 N5
  N6 N7 N8
  
  R[0] = [0, 0, 0, 0, 6, 0, 0, 0, 0]				   C[0] = [0, 9, 0, 0, 0, 0, 0, 0, 0]
  R[1] = [9, 0, 0, 0, 0, 0, 0, 0, 0]				   C[1] = [0, 0, 0, 0, 0, 1, 2, 0, 6]
  R[2] = [0, 0, 0, 2, 8, 3, 0, 0, 5]				   C[2] = [0, 0, 0, 5, 0, 0, 9, 3, 0]
  R[3] = [0, 0, 5, 0, 0, 0, 9, 0, 8]				   C[3] = [0, 0, 2, 0, 0, 3, 1, 5, 8]
  R[4] = [0, 0, 0, 0, 0, 0, 6, 1, 0]				   C[4] = [6, 0, 8, 0, 0, 0, 0, 0, 2]
  R[5] = [0, 1, 0, 3, 0, 0, 0, 0, 0]				   C[5] = [0, 0, 3, 0, 0, 0, 0, 0, 0]
  R[6] = [0, 2, 9, 1, 0, 0, 5, 0, 0]				   C[6] = [0, 0, 0, 9, 6, 0, 5, 4, 0]
  R[7] = [0, 0, 3, 5, 0, 0, 4, 0, 0]				   C[7] = [0, 0, 0, 0, 1, 0, 0, 0, 0]
  R[8] = [0, 6, 0, 8, 2, 0, 0, 0, 7]				   C[8] = [0, 0, 5, 8, 0, 0, 0, 0, 7]
  
  N[0] = [ R[0][0,1,2], R[1][0,1,2], R[2][0,1,2], C[0][0,1,2], C[1][0,1,2], C[2][0,1,2] ]
  N[1] = [ R[0][3,4,5], R[1][3,4,5], R[2][3,4,5], C[3][3,4,5], C[4][3,4,5], C[5][3,4,5] ]
  N[2] = [ R[0][6,7,8], R[1][6,7,8], R[2][6,7,8], C[6][6,7,8], C[7][6,7,8], C[8][6,7,8] ]
  
  N[3] = [ R[3][0,1,2], R[4][0,1,2], R[5][0,1,2], C[0][0,1,2], C[1][0,1,2], C[2][0,1,2] ]
  N[4] = [ R[3][3,4,5], R[4][3,4,5], R[5][3,4,5], C[3][3,4,5], C[4][3,4,5], C[5][3,4,5] ]
  N[5] = [ R[3][6,7,8], R[4][6,7,8], R[5][6,7,8], C[6][6,7,8], C[7][6,7,8], C[8][6,7,8] ]
  
  N[6] = [ R[6][0,1,2], R[7][0,1,2], R[8][0,1,2], C[0][0,1,2], C[1][0,1,2], C[2][0,1,2] ]
  N[7] = [ R[6][3,4,5], R[7][3,4,5], R[8][3,4,5], C[3][3,4,5], C[4][3,4,5], C[5][3,4,5] ]
  N[8] = [ R[6][6,7,8], R[7][6,7,8], R[8][6,7,8], C[6][6,7,8], C[7][6,7,8], C[8][6,7,8] ]

  
  '''

import sys
import numpy as np
import copy

# Creates the initial game state
class sudokuBoard:
	def __init__(self, listRows, listCols):
		self.rows = listRows
		self.cols = listCols
		self.cells = None
		self.grid = [[0 for x in range(len(self.rows))] for y in range(len(self.cols))]
		
		# Generate Game Grid
		for i in range(len(self.rows)):
			for j in range(len(self.rows)):
				self.grid[i][j] = self.rows[i][j]
		
		# Generate Cells
		nCell = [0 for i in range(9)]
		nCell[0] = self.rows[0][:3] + self.rows[1][:3] + self.rows[2][:3]
		nCell[1] = self.rows[0][3:6] + self.rows[1][3:6] + self.rows[2][3:6]
		nCell[2] = self.rows[0][6:] + self.rows[1][6:] + self.rows[2][6:]

		nCell[3] = self.rows[3][:3] + self.rows[4][:3] + self.rows[5][:3]
		nCell[4] = self.rows[3][3:6] + self.rows[4][3:6] + self.rows[5][3:6]
		nCell[5] = self.rows[3][6:] + self.rows[4][6:] + self.rows[5][6:]

		nCell[6] = self.rows[6][:3] + self.rows[7][:3] + self.rows[8][:3]
		nCell[7] = self.rows[6][3:6] + self.rows[7][3:6] + self.rows[8][3:6]
		nCell[8] = self.rows[6][6:] + self.rows[7][6:] + self.rows[8][6:]
				
		self.cells = nCell
			
	def getRows(self):
		return self.rows
		
	def getCols(self):
		return self.cols
	
	def getGrid(self):
		return self.grid
		
	def getCells(self):
		return self.cells				
	
	def getCellFromPos(self, r, c):
		if (r <= 2 and r >= 0):	#First row of cells
			if (c <= 2 and c >= 0):
				return self.cells[0]
			elif (c <= 5 and c >= 3):
				return self.cells[1]
			elif (c <= 8 and c >= 6):
				return self.cells[2]
		elif (r <= 5 and r >= 3):
			if (c <= 2 and c >= 0):
				return self.cells[3]
			elif (c <= 5 and c >= 3):
				return self.cells[4]
			elif (c <= 8 and c >= 6):
				return self.cells[5]
		elif (r <= 8 and r >= 6):
			if (c <= 2 and c >= 0):
				return self.cells[6]
			elif (c <= 5 and c >= 3):
				return self.cells[7]
			elif (c <= 8 and c >= 6):
				return self.cells[8]
	
	def getGameState(self):
		return (self.rows, self.cols, self.grid, self.cells)
		
	def generateNewGrid(self):
		# Uses rows to construct a new NxN game grid
		for i in range(len(self.rows)):
			for j in range(len(self.rows)):
				self.grid[i][j] = self.rows[i][j]

	def generateNewNCells(self):
		nCell = [0 for i in range(9)]
		nCell[0] = self.rows[0][:3] + self.rows[1][:3] + self.rows[2][:3]
		nCell[1] = self.rows[0][3:6] + self.rows[1][3:6] + self.rows[2][3:6]
		nCell[2] = self.rows[0][6:] + self.rows[1][6:] + self.rows[2][6:]

		nCell[3] = self.rows[3][:3] + self.rows[4][:3] + self.rows[5][:3]
		nCell[4] = self.rows[3][3:6] + self.rows[4][3:6] + self.rows[5][3:6]
		nCell[5] = self.rows[3][6:] + self.rows[4][6:] + self.rows[5][6:]

		nCell[6] = self.rows[6][:3] + self.rows[7][:3] + self.rows[8][:3]
		nCell[7] = self.rows[6][3:6] + self.rows[7][3:6] + self.rows[8][3:6]
		nCell[8] = self.rows[6][6:] + self.rows[7][6:] + self.rows[8][6:]
				
		self.cells = nCell
	
	def setNewGridValue(self, x, y, value):
		self.grid[x][y] = value
		
		#Re-evaluate the rows and columns
		for i in range(len(self.rows[x])):
			if i == y:
				self.rows[x][y] = value
		for j in range(len(self.cols[y])):
			if j == x:
				self.cols[y][x] = value
				
		self.generateNewNCells()
	
	
		
# Performs all actions on the game state.				
class agentCSP:
	def __init__(self, game):
		self.rows = game.getRows()
		self.cols = game.getCols()
		self.cells = game.getCells()
		self.gameState = game.getGameState()
		
		
	def isBoardFull(self, game):
		for i in range(len(game.getGameState())):
			for j in range(len(game.getGameState())):
				test = game.getGameState()
				if test[i][j] == '0':
					return False
		return True
		
	def isGoalState(self, game):
		# Check if board is filled
		cells = game.getCells()
		if self.isBoardFull(game):
			#Check Rows
			for i in range(len(game.rows)):
				#setRow = set(self.rows[i])
				checkRow = list(set(game.rows[i]))
				if (len(checkRow) != 9) or ('0' in checkRow):
					#print(checkRow)
					#print(len(checkRow))
					return False
			#Check Columns
			for j in range(len(game.cols)):
				#setCol = set(self.cols[j])
				checkCol = list(set(game.cols[j]))
				if (len(checkCol) != 9) or ('0' in checkCol):
					#print(checkCol)
					#print(len(checkCol))
					return False
			#Check Cells
			for k in range(len(cells)):
				#setCell = set(self.cells[k])
				setCell = set(cells[k])
				checkCell = list(setCell)
				if (len(checkCell) != 9) or ('0' in checkCell):
					#print(len(checkCell))
					return False
			
		else:
			return False
		
		return True
		
	def getDomainHelper(self, entry):
		# Takes in a single row, column, or cell as a parameter (called entry)
		# Forward Checking, gets domain of row or column
		U = {'1','2','3','4','5','6','7','8','9'}
		R_temp = []
		for i in range(len(entry)):
			if entry[i] != '0':
				R_temp.append(entry[i])		
				
		domain = list(U - set(R_temp))
		
		return domain
		
		
		
	def getDomain(self, x, y, game):
		R_temp = []
		C_temp = []
		N_temp = []
		domain = []
		row = game.getRows()
		#print(row[x])
		column = game.getCols()
		#print(column[y])
		
		C_temp = self.getDomainHelper(column[y])
		R_temp = self.getDomainHelper(row[x])
		N_temp = self.getDomainHelper(game.getCellFromPos(x,y))
		#print(C_temp)
		#print(R_temp)
		#print(N_temp)
		
		D_temp = C_temp + R_temp + N_temp
		#print(D_temp)
		
		for i in range(1,10):
			if D_temp.count(str(i)) == 3:
				domain.append(str(i))
		
		#print(domain)		
		return domain	
		
	def findNextEmptySpace(self, game):
		rows = game.getRows()
		space = len(rows)
		for i in range(space):
			for j in range(space):
				if rows[i][j] == '0':
					return (i,j)
				
	def findMostConstrainedSpace(self, game):
		rows = game.getRows()
		space = len(rows)
		for i in range(space):
			for j in range(space):
				if (len(self.getDomain(i,j,game)) == 1) and (rows[i][j] == '0'):
					return (i,j)
	
	def printBoard(self, game):
		rows = game.getRows()
		for i in range(len(rows)):
			print(rows[i])
	
	def searchCSP(self, game):
		while not (self.isGoalState(game)):
			posX, posY = self.findMostConstrainedSpace(game)
			val = self.getDomain(posX, posY, game)
			game.setNewGridValue(posX, posY, val[0])
			
		self.printBoard(game)
		return game
				
	
	
	'''
	def searchCSP(self, game):
		stack = []
		val = None
		posX, posY = self.findNextEmptySpace(game)
		initialX, initialY = (posX, posY)
		#print(game.getGrid())
		#print(posX, posY)
		lastVal = '0'
		
		while not (self.isGoalState(game)):
			print('searchCSP loop iteration')
			
			domain = self.getDomain(posX, posY, game)
			print(domain)
			
			if lastVal != '0':
				print('lastVal:')
				print(lastVal)
				if len(domain) > 1:
					#domain = domain.remove(lastVal)
					domain = lastDom.remove(lastVal)
				else:
					domain = []
					
			
			if len(domain) > 0:
				# Cast domain to ints:
				domain_temp = []
				for i in range(len(domain)):
					for j in range(1,10):
						if int(domain[i]) == j:
							domain_temp.append(j)
							
				val = str(min(domain_temp))
				
				stack.append((val, posX, posY, domain))
				print('Appending ' + val + ' to stack')
				#stack.append((val, posX, posY))
				game.setNewGridValue(posX, posY, val)
				
				posX, posY = self.findNextEmptySpace(game)
				lastVal = '0'
				
			elif len(domain) == 0:
				
				if len(stack) > 0:
					print(stack[-1])
				else:
					print('Empty Stack')
				lastVal, lastX, lastY, lastDom = stack.pop()
				#lastVal, lastX, lastY = stack.pop()
				game.setNewGridValue(lastX, lastY, '0')
				posX = lastX
				posY = lastY
				
			
		return game
		
		
	def test_CSP(self, game):
		if self.isGoalState(game) == True:
			return (True, game)
		
		posX, posY = self.findNextEmptySpace(game)
		
		for i in range(1,10):
			if i in self.getDomain(posX, posY, game):
				game.setNewGridValue(posX, posY, str(i))
				g2 = game.copy.deepcopy(game)
				
				if test_CSP(self, g2)[0] == True:
					return (True, game)
				
				game.setNewGridValue(posX, posY, '0')
		
		return False
		'''




# loadBoard reads in a .sdk file of a sudoku board, parses it
# into 
def loadBoard(filename):
	file = open(filename, 'r')
	rawBoardData = []
	
	# Read in line by line. Each row is a list element.
	for line in file:
		if line != '\n':
			rawBoardData.append(line)
	
	# Remove Header from .sdk
	rawBoardData.pop(0)
	
	# Creates two lists, one containing the rows, and another containing the columns
	row = []
	column = []
	for i in range(len(rawBoardData)):
		row.append(rawBoardData[i].split())
	for j in range(len(row)):
		tempColumn = []
		for k in range(len(row)):
			tempColumn.append(row[k][j])
		column.append(tempColumn)
		
	file.close()
	#print(column)
	
	return (row, column)
	
def CSP(x, y):
	G = sudokuBoard(x, y)
	Q = agentCSP(G)
	SolvedBoard = Q.searchCSP(G)
	

def main():
	x, y = loadBoard(sys.argv[1])
	return CSP(x, y)

main()
	