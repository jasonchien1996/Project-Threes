#pragma once
#include <array>
#include <iostream>
#include <iomanip>

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
	typedef uint32_t cell;
	typedef std::array<cell, 4> row;
	typedef std::array<row, 4> grid;
	typedef uint64_t data;
	typedef int reward;

public:
	board() : tile(), attr(0) {}
	board(const grid& b, data v = 0) : tile(b), attr(v) {}
	board(const board& b) = default;
	board& operator =(const board& b) = default;

	operator grid&() { return tile; }
	operator const grid&() const { return tile; }
	row& operator [](unsigned i) { return tile[i]; }
	const row& operator [](unsigned i) const { return tile[i]; }
	cell& operator ()(unsigned i) { return tile[i / 4][i % 4]; }
	const cell& operator ()(unsigned i) const { return tile[i / 4][i % 4]; }

	data info() const { return attr; }
	data info(data dat) { data old = attr; attr = dat; return old; }

public:
	bool operator ==(const board& b) const { return tile == b.tile; }
	bool operator < (const board& b) const { return tile <  b.tile; }
	bool operator !=(const board& b) const { return !(*this == b); }
	bool operator > (const board& b) const { return b < *this; }
	bool operator <=(const board& b) const { return !(b < *this); }
	bool operator >=(const board& b) const { return !(*this < b); }

public:
    int get_direct() const{ return direct; }
	cell get_tile(int i) { return tile[i / 4][i % 4]; }

	/**
	 * place a tile (index value) to the specific position (1-d form index)
	 * return 0 if the action is valid, or -1 if not
	 */
	reward place(unsigned pos, cell tile) {
		if (pos >= 16) return -1;
		if (tile != 1 && tile != 2 && tile != 3) return -1;
		operator()(pos) = tile;
		return 0;
	}

	/**
	 * apply an action to the board
	 * return the reward of the action, or -1 if the action is illegal
	 */
	reward slide(unsigned opcode) {
		switch (opcode & 0b11) {
		case 0: return slide_up();
		case 1: return slide_right();
		case 2: return slide_down();
		case 3: return slide_left();
		default: return -1;
		}
	}

	reward slide_left()
	{
		board prev = *this;
		reward score = 0;
		for (int r = 0; r < 4; r++)
        {
			auto& row = tile[r];
			int hold = 0, blank = 0;
			for (int c = 0; c < 4; c++)
            {
				int tile = row[c];
				if (tile == 0){blank = 1;}
                else
                {
                    if (hold == tile && hold > 2)
                    {
                        row[c-1] = ++tile;
                        //score = (1 << tile);
                        blank = 1;
                    }
					else if (abs((hold - tile)) == 1 && (hold + tile) == 3)
                    {
                        row[c-1] = 3;
                        //score += (1 << tile);
                        blank = 1;
                    }
                    else
                        hold = tile;
                }
                if(blank == 1)
                {
                    for(int i = c; i < 3; i++)
                        row[i] = row[i+1];
                    row[3] = 0;
                    break;
                }
			}
		}
		direct = 3;
		if(*this != prev)
		{
		    int a = 0, b = 0;
            for(int i = 0; i < 16; i++)
            {
                if(tile[i/4][i%4] > 2)
                {
                    int power = 1, n = tile[i/4][i%4]-2;
                    while(n--) power *= 3;
                    a += power;
                }
                if(prev.tile[i/4][i%4] > 2)
                {
                    int power = 1, n = prev.tile[i/4][i%4]-2;
                    while(n--) power *= 3;
                    b += power;
                }
            }
            score = a - b;
            return score;
		}
		return -1;
	}
	reward slide_right() {
		reflect_horizontal();
		reward score = slide_left();
		reflect_horizontal();
		direct = 1;
		return score;
	}
	reward slide_up() {
		rotate_right();
		reward score = slide_right();
		rotate_left();
		direct = 0;
		return score;
	}
	reward slide_down() {
		rotate_right();
		reward score = slide_left();
		rotate_left();
		direct = 2;
		return score;
	}

	void transpose() {
		for (int r = 0; r < 4; r++) {
			for (int c = r + 1; c < 4; c++) {
				std::swap(tile[r][c], tile[c][r]);
			}
		}
	}

	void reflect_horizontal() {
		for (int r = 0; r < 4; r++) {
			std::swap(tile[r][0], tile[r][3]);
			std::swap(tile[r][1], tile[r][2]);
		}
	}

	void reflect_vertical() {
		for (int c = 0; c < 4; c++) {
			std::swap(tile[0][c], tile[3][c]);
			std::swap(tile[1][c], tile[2][c]);
		}
	}

	/**
	 * rotate the board clockwise by given times
	 */
	void rotate(int r = 1) {
		switch (((r % 4) + 4) % 4) {
		default:
		case 0: break;
		case 1: rotate_right(); break;
		case 2: reverse(); break;
		case 3: rotate_left(); break;
		}
	}

	void rotate_right() { transpose(); reflect_horizontal(); } // clockwise
	void rotate_left() { transpose(); reflect_vertical(); } // counterclockwise
	void reverse() { reflect_horizontal(); reflect_vertical(); }

public:
	friend std::ostream& operator <<(std::ostream& out, const board& b) {
		out << "+------------------------+" << std::endl;
		for (auto& row : b.tile) {
			out << "|" << std::dec;
			for (auto t : row) out << std::setw(6) << ((1 << t) & -2u);
			out << "|" << std::endl;
		}
		out << "+------------------------+" << std::endl;
		return out;
	}

private:
	grid tile;
	data attr;
	int direct = -1;
};
