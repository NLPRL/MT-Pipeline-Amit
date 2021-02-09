from wxconv import WXC

con = WXC(order='utf2wx')
con1 = WXC(order='wx2utf', lang='hin')

print(con.convert('लेस।'))
print(con1.convert('esehI'))