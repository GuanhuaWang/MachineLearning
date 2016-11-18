
# Divides a set on a specific column. Can handle numeric or nominal values
def divideset(rows,column,value):
   # Make a function that tells us if a row is in the first group (true) or the second group (false)
   split_function=None
   if isinstance(value,int) or isinstance(value,float): # check if the value is a number i.e int or float
      split_function=lambda row:row[column]>=value
   else:
      split_function=lambda row:row[column]==value
   
   # Divide the rows into two sets and return them
   set1=[row for row in rows if split_function(row)]
   set2=[row for row in rows if not split_function(row)]
   return (set1,set2)


def printtree(tree,indent=''):
   if tree.results!=None:
      print(str(tree.results))
   else:
      print(str(tree.col)+':'+str(tree.value)+'? ')
      print indent+'T->',
      printtree(tree.tb,indent+'  ')
      print indent+'F->',
      printtree(tree.fb,indent+'  ')

class decisionnode:
   def __init__(self,col=-1,value=None,results=None,tb=None,fb=None):
      self.col=col
      self.value=value
      self.results=results
      self.tb=tb
      self.fb=fb
