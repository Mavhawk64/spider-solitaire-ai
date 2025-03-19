import SpiderSolitaire as ss

game = ss.SpiderSolitaire(4, 0)
game.display_board()

while not game.has_won():
    try:
        play = input("Move, Draw, or Undo? (m/d/u): ").lower()
        if play == "d":
            game.draw_cards()
            print("\nAfter drawing cards:")
            game.display_board()
            continue
        elif play == "u":
            if game.undo():
                print("\nUndo successful.")
            else:
                print("\nNo moves to undo.")
            game.display_board()
            continue
        elif play == "e" or play == "^c":
            break
        elif play == "m":
            col1 = int(input("Enter the source column: "))
            col2 = int(input("Enter the target column: "))
            bundle = int(input("Enter the bundle length: "))
            if game.move_bundle(col1, col2, bundle):
                print("\nMove successful.")
            else:
                print("\nMove failed.")
            game.display_board()
            print("\nCurrent game state:")
            print(game.get_game_state())
        else:
            print(
                "Invalid input. Please enter 'm' for move, 'd' for draw, or 'u' for undo."
            )
    except Exception as e:
        print(f"An error occurred: {e}")
        break

if game.has_won():
    print("Congratulations, you've won!")
