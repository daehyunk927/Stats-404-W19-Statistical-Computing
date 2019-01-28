import random

theBoard = {'top-L': ' ', 'top-M': ' ', 'top-R': ' ',
            'mid-L': ' ', 'mid-M': ' ', 'mid-R': ' ',
            'low-L': ' ', 'low-M': ' ', 'low-R': ' '}

def printBoard(board):
    print(board['top-L'] + '|' + board['top-M'] + '|' + board['top-R'])
    print('-+-+-')
    print(board['mid-L'] + '|' + board['mid-M'] + '|' + board['mid-R'])
    print('-+-+-')
    print(board['low-L'] + '|' + board['low-M'] + '|' + board['low-R'])


def isWon(board):
    mark = ['X','O']
    # horizontal
    if board['top-L'] in mark and board['top-L'] == board['top-M'] == board['top-R']:
        return board['top-L']
    if board['mid-L'] in mark and board['mid-L'] == board['mid-M'] == board['mid-R']:
        return board['mid-L']
    if board['low-L'] in mark and board['low-L'] == board['low-M'] == board['low-R']:
        return board['low-L']
    # diagonal
    if board['top-L'] in mark and board['top-L'] == board['mid-M'] == board['low-R']:
        return board['top-L']
    if board['top-R'] in mark and board['top-R'] == board['mid-M'] == board['low-L']:
        return board['top-R']
    # vertical
    if board['top-L'] in mark and board['top-L'] == board['mid-L'] == board['low-L']:
        return board['top-L']
    if board['top-M'] in mark and board['top-M'] == board['mid-M'] == board['low-M']:
        return board['top-M']
    if board['top-R'] in mark and board['top-R'] == board['mid-R'] == board['low-R']:
        return board['top-R']

    return ' '
#pdb.set_trace()

turn = 'X'

for i in range(9):
    print()
    printBoard(theBoard)
    location = random.choice(list(theBoard))
    while theBoard[location] != ' ':
        location = random.choice(list(theBoard))
    theBoard[location] = turn

    if isWon(theBoard) != ' ':
        break

    if turn == 'X':
        turn = 'O'
    else:
        turn = 'X'

print()
printBoard(theBoard)
if isWon(theBoard) != ' ':
    print('game is over and player ' + isWon(theBoard) + ' has won.')
else:
    print('game is tied.')