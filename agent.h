#pragma once
#include <string>
#include <random>
#include <sstream>
#include <map>
#include <type_traits>
#include <algorithm>
#include "board.h"
#include "action.h"
#include "weight.h"
#include <fstream>
#include <math.h>

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

    int hint = 0;
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
 * base agent for agents with weight tables
 */
class weight_agent : public agent {
public:
	weight_agent(const std::string& args = "") : agent(args) {
		//if (meta.find("init") != meta.end()) // pass init=... to initialize the weight
			init_weights(meta["init"]);
		if (meta.find("load") != meta.end()) // pass load=... to load from a specific file
			load_weights(meta["load"]);
	}
	virtual ~weight_agent() {
		if (meta.find("save") != meta.end()) // pass save=... to save to a specific file
			save_weights(meta["save"]);
	}

protected:
	virtual void init_weights(const std::string& info) {
		net.emplace_back(136687500); // create an empty weight table with size 15**6*11
		net.emplace_back(136687500); // now net.size() == 2; net[0].size() == 65536; net[1].size() == 65536
	}
	virtual void load_weights(const std::string& path) {
		std::ifstream in(path, std::ios::in | std::ios::binary);
		if (!in.is_open()) std::exit(-1);
		uint32_t size;
		in.read(reinterpret_cast<char*>(&size), sizeof(size));
		net.resize(size);
		for (weight& w : net) in >> w;
		in.close();
	}
	virtual void save_weights(const std::string& path) {
		std::ofstream out(path, std::ios::out | std::ios::binary | std::ios::trunc);
		if (!out.is_open()) std::exit(-1);
		uint32_t size = net.size();
		out.write(reinterpret_cast<char*>(&size), sizeof(size));
		for (weight& w : net) out << w;
		out.close();
	}

protected:
	std::vector<weight> net;
};

/**
 * base agent for agents with a learning rate
 */
class learning_agent : public weight_agent {
public:
	learning_agent(const std::string& args = "") : weight_agent(args), alpha(0.4f) {
		if (meta.find("alpha") != meta.end())
			alpha = float(meta["alpha"]);
	}
	virtual ~learning_agent() {}

protected:
	float alpha;
};

/**
 * random environment
 * add a new random tile to an empty cell
 * 2-tile: 90%
 * 4-tile: 10%
 */
class rndenv : public random_agent{
public:
    void reset(){
        bag = {1,1,1,1,2,2,2,2,3,3,3,3};
        std::shuffle(bag.begin(), bag.end(), engine);
        bonus.clear();
        bonus_max = 0;
        num_bag = 0;
        num_bonus = 0;
        hint = 0;
    }

	rndenv(const std::string& args = "") : random_agent("name=random role=environment " + args), bag({1,1,1,1,2,2,2,2,3,3,3,3}),
		space({ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 }), op_0({12, 13, 14, 15}), op_1({0, 4, 8, 12}), op_2({0, 1, 2, 3}), op_3({3, 7, 11, 15}){}

    void generate_hint(const board& after){
        int new_hint;
        int b_max = after.get_board_max();
        if(bag.empty()){
            bag = {1,1,1,1,2,2,2,2,3,3,3,3};
            std::shuffle(bag.begin(), bag.end(), engine);
        }
        //exist 48-tile
        if(b_max > 6){
            if(b_max-3 > bonus_max){
                bonus_max = b_max-3;
                bonus.push_back(bonus_max);
            }
            std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
            std::uniform_int_distribution<> pro(1, 21);
            //bonus
            if(pro(gen) == 1){
                float ratio = (num_bonus+1)/(num_bonus+num_bag+1);
                if(ratio <= 1/21){
                    std::uniform_int_distribution<> dis(0, bonus.size()-1);
                    new_hint = bonus[dis(gen)];
                    num_bonus++;
                }
                else{
                    new_hint = bag.back();
                    bag.pop_back();
                    num_bag++;
                }
            }
            //no bonus
            else{
                new_hint = bag.back();
                bag.pop_back();
                num_bag++;
            }
        }
        //non-exist 48-tile
        else{
            new_hint = bag.back();
            bag.pop_back();
            num_bag++;
        }
        hint = new_hint;
    }

	virtual action take_action(const board& after){
        int last = after.get_direct();
        int h = hint;
        /*std::cout<<"before evil\n"<<after;
        std::cout<<"hint"<<hint<<std::endl;
        for (auto i = bag.begin(); i != bag.end(); ++i)
            std::cout << *i << ' ';
        std::cout<<"\n";*/

        if(last == -1){
            std::shuffle(space.begin(), space.end(), engine);
            if(h == 0){
                h = bag.back();
                bag.pop_back();
                num_bag++;
            }
            for (int pos : space){
                if (after(pos) != 0) continue;
                generate_hint(after);
                return action::place(pos, h);
            }
        }
        else if(last == 0) op = op_0;
        else if(last == 1) op = op_1;
        else if(last == 2) op = op_2;
        else if(last == 3) op = op_3;
        std::shuffle(op.begin(), op.end(), engine);
        for(int pos : op){
            if(after(pos) != 0) continue;
            generate_hint(after);
            return action::place(pos, h);
        }
        return action();
	}

private:
	std::array<int, 16> space;
    std::vector<int> bag;
    std::vector<int> bonus;
    int bonus_max = 0;
    int num_bag = 0;
    int num_bonus = 0;
    std::array<int, 4> op;
	std::array<int, 4> op_0;
	std::array<int, 4> op_1;
	std::array<int, 4> op_2;
	std::array<int, 4> op_3;
    std::random_device rd;//Will be used to obtain a seed for the random number engine
};

/**
 * td0 player
 * select a legal action
 */
class player : public learning_agent {
public:
	player(const std::string& args = "") : learning_agent("name=learning role=player " + args){}

    int find_index(int j, board as) {
        int index = 0;
        for(int k = 0; k < 6; k++) {
            int power = 1, n = k;
            while(n--) power *=15;
            power *= as.operator()(pattern[j][k]);
            index += power;
        }
        index = 11390625 * hint + index;
        return index;
    }

	virtual action take_action(const board &before) {
        /*std::cout<<"before play\n"<<before;
        std::cout<<"hint"<<hint<<std::endl;*/

        int op, index, valid = 0, t_r, reward;
        float t = -999999, direction_value;
        std::array<std::array<int, 32>, 4> key;
        board as = before;
        // find best action
        for(int i = 0; i < 4; i++){
            direction_value = 0;
            as = before;
            reward = as.slide(i); // r
            if(reward != -1){
                valid = 1;
                direction_value += reward;
                for(int j = 0; j < 32; j++){
                    index = find_index(j,as);
                    key[i][j] = index;
                    if(j < 16)
                        direction_value += net[0][index]; // v(as)
                    else
                        direction_value += net[1][index];
                }
                if(direction_value > t){
                    t = direction_value;
                    op = i;
                    t_r = reward;
                }
            }
        }
        //not terminal
        if(valid == 1) {
            state_key.push_back(key[op]);
            r.push_back(t_r);
            return action::slide(op);
        }
        //terminal
        for(int j = 0; j < 32; j++){
            index = find_index(j,as);
            key[0][j] = index;
        }
        state_key.push_back(key[0]);
        r.push_back(0);
        return action();
    }

    void training() {
        std::array<int, 32> as_key;
        float sum = 0, v_as, amend;
        r.push_back(0);
        while(!state_key.empty()){
            as_key = state_key.back();
            v_as = sum;
            sum = 0;
            for(int j = 0; j < 32; j++){
                if(j<16)
                    sum += net[0][as_key[j]];
                else
                    sum += net[1][as_key[j]];
            }
            amend = alpha * (r.back() + v_as - sum) / 32;
            sum = 0;
            for(int i = 0; i < 32; i++){
                if(i < 16){
                    net[0][as_key[i]] += amend;
                    sum += net[0][as_key[i]];
                }
                else{
                    net[1][as_key[i]] += amend;
                    sum += net[1][as_key[i]];
                }
            }
            state_key.pop_back();
            r.pop_back();
        }
        r.pop_back();
    }

private:
	std::vector<int> r;
    std::vector<std::array<int, 32>> state_key;
    const int pattern[32][6]={{ 0, 4, 8, 9,12,13},
                        { 1, 5, 9,10,13,14},
                        { 3, 2, 1, 5, 0, 4},
                        { 7, 6, 5, 9, 4, 8},
                        {14,10, 6, 5, 2, 1},
                        {15,11, 7, 6, 3, 2},
                        { 8, 9,10, 6,11, 7},
                        {12,13,14,10,15,11},
                        { 0, 1, 2, 6, 3, 7},
                        { 4, 5, 6,10, 7,11},
                        { 2, 6,10, 9,14,13},
                        { 3, 7,11,10,15,14},
                        {11,10, 9, 5, 8, 4},
                        {15,14,13, 9,12, 8},
                        {12, 8, 4, 5, 0, 1},
                        {13, 9, 5, 6, 1, 2},//
                        { 1, 2, 5, 6, 9,10},//
                        { 2, 3, 6, 7,10,11},
                        { 7,11, 6,10, 5, 9},
                        {11,15,10,14, 9,13},
                        {14,13,10, 9, 6, 5},
                        {13,12, 9, 8, 5, 4},
                        { 8, 4, 9, 5,10, 6},
                        { 4, 0, 5, 1, 6, 2},
                        {13,14, 9,10, 5, 6},
                        {14,15,10,11, 6, 7},
                        {11, 7,10, 6, 9, 5},
                        { 7, 3, 6, 2, 5, 1},
                        { 2, 1, 6, 5,10, 9},
                        { 1, 0, 5, 4, 9, 8},
                        { 4, 8, 5, 9, 6,10},
                        { 8,12, 9,13,10,14}};
};
