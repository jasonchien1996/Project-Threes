#pragma once
#include <array>
#include <iostream>
#include <iomanip>
#include <vector>

/**
 * array-based board for 2048
 *
 * index (1-d form):
 *  (0)  (1)  (2)  (3)
 *  (4)  (5)  (6)  (7)
 *  (8)  (9) (10) (11)
 * (12) (13) (14) (15)
 *
 */
class board {
public:
	typedef uint8_t cell;
	typedef std::array<cell, 4> row;
	typedef std::array<row, 4> grid;
	typedef int reward;

public:
	board() : tile(), grade({0,0,0,0,18,63,216,729,2430,8019,26244,85293,275562,885735,2834352,9034497}) {}
	board(const grid& b) : tile(b), grade({0,0,0,0,18,63,216,729,2430,8019,26244,85293,275562,885735,2834352,9034497}) {}
	board(const board& b) = default;
	board& operator =(const board& b) = default;

	operator grid&() { return tile; }
	operator const grid&() const { return tile; }
	row& operator [](unsigned i) { return tile[i]; }
	const row& operator [](unsigned i) const { return tile[i]; }
	cell& operator ()(unsigned i) { return tile[i / 4][i % 4]; }
	const cell& operator ()(unsigned i) const { return tile[i / 4][i % 4]; }

public:
	bool operator ==(const board& b) const { return tile == b.tile; }
	bool operator !=(const board& b) const { return !(*this == b); }

public:
	/**
	 * place a tile (index value) to the specific position (1-d form index)
	 * return 0 if the action is valid, or -1 if not
	 */
	reward place(unsigned pos, cell tile){
		if(pos >= 16) return -1;
		if(max > 6){
            if(tile > abs(max-3)) return -1;
		}
    else{
        if(tile != 1 && tile != 2 && tile != 3) return -1;
    }
		operator()(pos) = tile;
		return 0;
	}

	/**
	 * apply an action to the board
	 * return the reward of the action, or -1 if the action is illegal
	 */
	reward slide(unsigned opcode){
		switch (opcode & 0b11) {
		case 0: return slide_up();
		case 1: return slide_right();
		case 2: return slide_down();
		case 3: return slide_left();
		default: return -1;
		}
	}

	reward slide_left(){
		board prev = *this;
		reward score = 0;
		for(int r = 0; r < 4; ++r){
			row& row = tile[r];
			int hold = 0, blank = 0;
			for(int c = 0; c < 4; ++c){
				int tile = row[c];
				if(tile == 0){blank = 1;}
                else{
                    if(hold == tile && hold > 2){
                        row[c-1] = ++tile;
                        blank = 1;
                        score += grade[row[c-1]];//3^(row[c-1]-3) * (row[c-1] + 2);
                        if(row[c-1] > max) max = row[c-1];
                    }
					else if(abs((hold - tile)) == 1 && (hold + tile) == 3){
                        row[c-1] = 3;
                        blank = 1;
                        score += 9;
                        if(row[c-1] > max) max = row[c-1];
                    }
                    else
                        hold = tile;
                }
                if(blank == 1){
                    for(int i = c; i < 3; i++) row[i] = row[i+1];
                    row[3] = 0;
                    break;
                }
			}
		}
		last = 3;
		if(*this != prev) return score;
		return -1;
	}
	reward slide_right(){
		reflect_horizontal();
		reward score = slide_left();
		reflect_horizontal();
		last = 1;
		return score;
	}
	reward slide_up(){
		rotate_right();
		reward score = slide_right();
		rotate_left();
		last = 0;
		return score;
	}
	reward slide_down(){
		rotate_right();
		reward score = slide_left();
		rotate_left();
		last = 2;
		return score;
	}

	void transpose(){
		for(int r = 0; r < 4; ++r){
			for(int c = r + 1; c < 4; c++){
				std::swap(tile[r][c], tile[c][r]);
			}
		}
	}

	void reflect_horizontal(){
		for(int r = 0; r < 4; ++r){
			std::swap(tile[r][0], tile[r][3]);
			std::swap(tile[r][1], tile[r][2]);
		}
	}

	void reflect_vertical(){
		for(int c = 0; c < 4; ++c){
			std::swap(tile[0][c], tile[3][c]);
			std::swap(tile[1][c], tile[2][c]);
		}
	}

	void rotate_right() { transpose(); reflect_horizontal(); } // clockwise
	void rotate_left() { transpose(); reflect_vertical(); } // counterclockwise

public:
	friend std::ostream& operator <<(std::ostream& out, const board& b){
		out << "+------------------------+" << std::endl;
		for(auto& row : b.tile){
			out << "|" << std::dec;

			for(auto t : row){
                    int k = (t > 3) ? (1 << (t-3) & -2u)*3 : t;
                    out << std::setw(6) << k;
			}
			out << "|" << std::endl;
		}
		out << "+------------------------+" << std::endl;
		return out;
	}

public:
    char type;
    int hint;
   	int last = -1;
    int max = 0;
    std::array<float,3> bag;
    
private:
	grid tile;
    std::array<int,16> grade;
};
