def pr(texto):
    lista = ("adios","chau","salir")
    if texto in lista:
        print("finalizado")
        return
    else:
        print("sigue")
        pr(texto)

pr("adios")
