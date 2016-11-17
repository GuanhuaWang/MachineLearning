
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