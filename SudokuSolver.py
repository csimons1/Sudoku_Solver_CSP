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

# Creates the initial game state
class sudokuBoard:
	def __init__(self, listRows, listCols):
		self.rows = listRows
		self.cols = listCols
		
		# Generate Game Grid?
		self.grid = [[0 for x in range(len(self.rows))] for y in range(len(self.cols))]
		for i in range(len(self.rows)):
			for j in range(len(self.rows)):
				self.grid[i][j] = self.rows[i][j]
		
		# Generate Cells?
		
		# Might need to make this a method, and not make it into a multidimensional array. keep as list,
		# the checking algorithm will be easier, and modifications to the gamestate made by the csp agent
		# should use methods in this class to convert the changes into cells/rows/cols
		nCell = [0 for i in range(9)]
		nCell[0] = str(self.rows[0][:3] + self.rows[1][:3] + self.rows[2][:3])
		nCell[1] = str(self.rows[0][3:6] + self.rows[1][3:6] + self.rows[2][3:6])
		nCell[2] = str(self.rows[0][6:] + self.rows[1][6:] + self.rows[2][6:])

		nCell[3] = str(self.rows[3][:3] + self.rows[4][:3] + self.rows[5][:3])
		nCell[4] = str(self.rows[3][3:6] + self.rows[4][3:6] + self.rows[5][3:6])
		nCell[5] = str(self.rows[3][6:] + self.rows[4][6:] + self.rows[5][6:])

		nCell[6] = str(self.rows[6][:3] + self.rows[7][:3] + self.rows[8][:3])
		nCell[7] = str(self.rows[6][3:6] + self.rows[7][3:6] + self.rows[8][3:6])
		nCell[8] = str(self.rows[6][6:] + self.rows[7][6:] + self.rows[8][6:])
				
		self.cells = nCell
			
	def getRows(self):
		return self.rows
		
	def getCols(self):
		return self.cols
	
	def getGrid(self):
		return self.grid
		
	def getCells(self):
		return self.cells
	
	def getGameState(self):
		return (self.rows, self.cols, self.grid, self.cells)
		
	def generateNewGrid(self):
		for i in range(len(self.rows)):
			for j in range(len(self.rows)):
				self.grid[i][j] = self.rows[i][j]

	def generateNewNCells(self):
		nCell = [0 for i in range(9)]
		nCell[0] = str(self.rows[0][:3] + self.rows[1][:3] + self.rows[2][:3])
		nCell[1] = str(self.rows[0][3:6] + self.rows[1][3:6] + self.rows[2][3:6])
		nCell[2] = str(self.rows[0][6:] + self.rows[1][6:] + self.rows[2][6:])

		nCell[3] = str(self.rows[3][:3] + self.rows[4][:3] + self.rows[5][:3])
		nCell[4] = str(self.rows[3][3:6] + self.rows[4][3:6] + self.rows[5][3:6])
		nCell[5] = str(self.rows[3][6:] + self.rows[4][6:] + self.rows[5][6:])

		nCell[6] = str(self.rows[6][:3] + self.rows[7][:3] + self.rows[8][:3])
		nCell[7] = str(self.rows[6][3:6] + self.rows[7][3:6] + self.rows[8][3:6])
		nCell[8] = str(self.rows[6][6:] + self.rows[7][6:] + self.rows[8][6:])
				
		self.cells = nCell
	
	def setNewGridValue(self, x, y, value)
		self.grid[x][y] = value
		
		#Re-evaluate the rows and columns
		for i in range(len(self.rows[x])):
			if i == y:
				self.rows[x][y] = value
		for j in range(len(self.cols[y])):
			if j == x:
				self.cols[x]
		
# Performs all actions on the game state.				
class agentCSP:
	def __init__(self, game):
		self.rows = game.getRows()
		self.cols = game.getCols()
		self.cells = game.getCells()
		self.gameState = game.getGameState()
		
		
	def isBoardFull(self):
		for i in range(len(self.gameState)):
			for j in range(len(self.gameState)):
				if self.gameState[i][j] == 0:
					return False
		return True
		
	def isGoalState(self, game):
		# Check if board is filled
		if isBoardFull(game):
			#Check Rows
			for i in range(len(self.rows)):
				setRow = set(self.rows[i])
				checkRow = list(setRow)
				if len(checkRow) != 9:
					return False
			#Check Columns
			for j in range(len(self.cols)):
				setCol = set(self.cols[j])
				checkCol = list(setCol)
				if len(checkCol) != 9:
					return False
			#Check Cells - NYI
			for k in range(len(self.cells)):
				setCell = set(self.cells[k])
				checkCell = list(setCell)
				if len(checkCell) != 9:
					return False
		else:
			return False
		
		return True
		
		
		
	
		
		




# loadBoard reads in a .sdk file of a sudoku board, parses it
# into 
def loadBoard(filename):
	file = open(filename, 'r')
	rawBoardData = []
	
	# Read in line by line. Each row is a list element.
	for line in file:
		rawBoardData.append(line)
	
	# Remove Header from .sdk
	rawBoardData.pop(0)
	
	# Creates two lists, one containing the rows, and another containing the columns
	row = []
	column = []
	for i in range(len(rawBoard)):
		row.append(rawBoard[i].split())
	for j in range(len(row)):
		tempColumn = []
		for k in range(len(row)):
			tempColumn.append(row[k][j])
		column.append(tempColumn)
		
	# Do N Cells need to be constructed?
	# Fuck it, don't forget to remove this comment
	# Cells are created by rows, not columns.
	'''
	nCell = [0 for i in range(9)]
	nCell[0] = [row[0][:3], row[1][:3], row[2][:3]]
	nCell[1] = [row[0][3:6], row[1][3:6], row[2][3:6]]
	nCell[2] = [row[0][6:], row[1][6:], row[2][6:]]

	nCell[3] = [row[3][:3], row[4][:3], row[5][:3]]
	nCell[4] = [row[3][3:6], row[4][3:6], row[5][3:6]]
	nCell[5] = [row[3][6:], row[4][6:], row[5][6:]]

	nCell[6] = [row[6][:3], row[7][:3], row[8][:3]]
	nCell[7] = [row[6][3:6], row[7][3:6], row[8][3:6]]
	nCell[8] = [row[6][6:], row[7][6:], row[8][6:]]
	
	return (row, column, nCells)
	'''
	return (row, column)
	
def CSP(x, y):
	G = sudokuBoard(x, y)
	Q = agentCSP(G)

def main():
	x, y = loadBoard(sys.argv[1])
	return CSP(x, y)
	