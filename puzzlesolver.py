
class Puzzle:

    def __init__(self,X):
        self.X=X

    def checkRow(self,num,row):
        PuzzleRow=self.X[row]
        if num in PuzzleRow:
            return False
        return True

    def checkCol(self,num,col):
        PuzzleCol=self.X[:,col]
        if num in PuzzleCol:
            return False
        return True

    def checkBox(self,num,si,sj):
        boxr=si/3
        boxc=sj/3

        boxr=boxr*3
        boxc=boxc*3
        for i in range(boxr,boxr+3):
            for j in range(boxc,boxc+3):
                if self.X[i][j]==num:
                    return False
        return True


    def findNext(self,i,j):
        while i < 9:
            while j < 9:
                if self.X[i][j] == 0:
                    return i,j
                j+=1
            j=0
            i+=1
        return None,None


    def help(self,i,j):
        for k in range(1,10):
            if self.checkRow(k,i) and self.checkCol(k,j) and self.checkBox(k,i,j):
                self.X[i][j]=k
                ni,nj=self.findNext(i,j+1)
                if ni == None:
                    return True
                else:
                    if self.help(ni,nj):
                        return True
                    else:
                        self.X[i][j]=0
        return False


    def solve(self):
        i,j=self.findNext(0,0)
        if i == None:
            return True

        if self.help(i,j):
            print 'Solved'
            return True
        else:
            print 'Cant Solve'
            return False




