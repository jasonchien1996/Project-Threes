#pragma once
#include <string>
#include <random>
#include <sstream>
#include <map>
#include <type_traits>
#include <algorithm>
#include "board.h"
#include "action.h"

class agent {
public:
	agent(const std::string& args = "") {
		std::stringstream ss("name=unknown role=unknown " + args);
		for (std::string pair; ss >> pair; ) {
			std::string key = pair.substr(0, pair.find('='));
			std::string value = pair.substr(pair.find('=') + 1);
			meta[key] = { value };
		}
	}
	virtual ~agent() {}
	virtual void open_episode(const std::string& flag = "") {}
	virtual void close_episode(const std::string& flag = "") {}
	virtual action take_action(const board& b) { return action(); }
	virtual bool check_for_win(const board& b) { return false; }

public:
	virtual std::string property(const std::string& key) const { return meta.at(key); }
	virtual void notify(const std::string& msg) { meta[msg.substr(0, msg.find('='))] = { msg.substr(msg.find('=') + 1) }; }
	virtual std::string name() const { return property("name"); }
	virtual std::string role() const { return property("role"); }

protected:
	typedef std::string key;
	struct value {
		std::string value;
		operator std::string() const { return value; }
		template<typename numeric, typename = typename std::enable_if<std::is_arithmetic<numeric>::value, numeric>::type>
		operator numeric() const { return numeric(std::stod(value)); }
	};
	std::map<key, value> meta;
};

class random_agent : public agent {
public:
	random_agent(const std::string& args = "") : agent(args) {
		if (meta.find("seed") != meta.end())
			engine.seed(int(meta["seed"]));
	}
	virtual ~random_agent() {}

protected:
	std::default_random_engine engine;
};

/**
 * random environment
 * add a new random tile to an empty cell
 * 2-tile: 90%
 * 4-tile: 10%
 */
class rndenv : public random_agent {
public:
    void reset_bag(){bag = {1,2,3};}
	rndenv(const std::string& args = "") : random_agent("name=random role=environment " + args),
		space({ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 }), op_0({12, 13, 14, 15}), op_1({0, 4, 8, 12}), op_2({0, 1, 2, 3}), op_3({3, 7, 11, 15}){}

	virtual action take_action(const board& after)
	{

        int init=after.get_direct();
        switch(init)
        {
            case -1:
                std::shuffle(space.begin(), space.end(), engine);
                for (int pos : space)
                {
                    if (after(pos) != 0) continue;
                    if (bag.empty())
                    {
                        bag = {1,2,3};
                        std::shuffle(bag.begin(), bag.end(), engine);
                    }
                    board::cell tile = bag.back();
                    bag.pop_back();
                    //std::cout<<"no slide"<<pos<<','<<tile<<std::endl;
                    return action::place(pos, tile);
                }
                break;
            case 0:
                std::shuffle(op_0.begin(), op_0.end(), engine);
                for (int pos : op_0)
                {
                    if (after(pos) != 0) continue;
                    if (bag.empty())
                    {
                        bag = {1,2,3};
                        std::shuffle(bag.begin(), bag.end(), engine);
                    }
                    board::cell tile = bag.back();
                    bag.pop_back();
                    //std::cout<<'u'<<pos<<','<<tile<<std::endl;
                    return action::place(pos, tile);
                }
                break;
            case 1:
                std::shuffle(op_1.begin(), op_1.end(), engine);
                for (int pos : op_1)
                {
                    if (after(pos) != 0) continue;
                    if (bag.empty())
                    {
                        bag = {1,2,3};
                        std::shuffle(bag.begin(), bag.end(), engine);
                    }
                    board::cell tile = bag.back();
                    bag.pop_back();
                    //std::cout<<'r'<<pos<<','<<tile<<std::endl;
                    return action::place(pos, tile);
                }
                break;
            case 2:
                std::shuffle(op_2.begin(), op_2.end(), engine);
                for (int pos : op_2)
                {
                    //std::cout<<after(2)<<std::endl;
                    if (after(pos) != 0) continue;
                    if (bag.empty())
                    {
                        bag = {1,2,3};
                        std::shuffle(bag.begin(), bag.end(), engine);
                    }
                    board::cell tile = bag.back();
                    bag.pop_back();
                    //std::cout<<'d'<<pos<<','<<tile<<std::endl;
                    return action::place(pos, tile);
                }
                break;
            case 3:
                std::shuffle(op_3.begin(), op_3.end(), engine);
                for (int pos : op_3)
                {
                    if (after(pos) != 0) continue;
                    if (bag.empty())
                    {
                        bag = {1,2,3};
                        std::shuffle(bag.begin(), bag.end(), engine);
                    }
                    board::cell tile = bag.back();
                    bag.pop_back();
                    //std::cout<<'l'<<pos<<','<<tile<<std::endl;
                    return action::place(pos, tile);
                }
                break;
            default:
                break;
        }
        //std::cout<<"action"<<std::endl;
        return action();
	}

private:
	std::array<int, 16> space;
    std::vector<int> bag;
	std::array<int, 4> op_0;
	std::array<int, 4> op_1;
	std::array<int, 4> op_2;
	std::array<int, 4> op_3;
};

/**
 * dummy player
 * select a legal action randomly
 */
class player : public random_agent {
public:
	player(const std::string& args = "") : random_agent("name=dummy role=player " + args),
		opcode({ 0, 1, 2, 3 }) {}

	virtual action take_action(const board& before) {
		std::shuffle(opcode.begin(), opcode.end(), engine);
		for (int op : opcode) {
			board::reward reward = board(before).slide(op);
			if (reward != -1) return action::slide(op);
		}
		return action();
	}

private:
	std::array<int, 4> opcode;
};
