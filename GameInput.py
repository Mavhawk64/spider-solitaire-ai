import SpiderSolitaire as ss



game = ss.SpiderSolitaire(4,0)
game.display_board()

while not game.has_won():
    try:
        play = input("Move or Draw? (m/d) ")
        if play == 'd':
            game.draw_cards()
            print('\n')
            game.display_board()
            continue
        if play == 'e' or play == '^C':
            break
        col1 = int(input("Enter the source column: "))
        col2 = int(input("Enter the target column: "))
        bundle = int(input("Enter the bundle length: "))
        game.move_bundle(col1,col2,bundle)
        print('\n')
        game.display_board()
    except:
        break