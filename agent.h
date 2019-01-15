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

class agent{
public:
	agent(const std::string& args = "") : bag({1,1,1,1,2,2,2,2,3,3,3,3}),
        op_0({12, 13, 14, 15}), op_1({0, 4, 8, 12}), op_2({0, 1, 2, 3}), op_3({3, 7, 11, 15}){
		std::stringstream ss("name=unknown role=unknown " + args);
		for(std::string pair; ss >> pair; ){
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

public:
    int hint = 0;
    std::vector<int> bag;
  
protected:
	typedef std::string key;
	struct value{
		std::string value;
		operator std::string() const { return value; }
		template<typename numeric, typename = typename std::enable_if<std::is_arithmetic<numeric>::value, numeric>::type>
		operator numeric() const { return numeric(std::stod(value)); }
	};
	std::map<key, value> meta;
	std::array<int, 4> op_0;
	std::array<int, 4> op_1;
	std::array<int, 4> op_2;
	std::array<int, 4> op_3;
};

class random_agent : public agent{
public:
	random_agent(const std::string& args = "") : agent(args){
		if(meta.find("seed") != meta.end())
			engine.seed(int(meta["seed"]));
	}
	virtual ~random_agent() {}

protected:
	std::default_random_engine engine;
};

/**
 * base agent for agents with weight tables
 */
class weight_agent : public agent{
public:
	weight_agent(const std::string& args = "") : agent(args){
		//if (meta.find("init") != meta.end()) // pass init=... to initialize the weight
			init_weights(meta["init"]);
		if(meta.find("load") != meta.end()) // pass load=... to load from a specific file
			load_weights(meta["load"]);
	}
	virtual ~weight_agent(){
		if(meta.find("save") != meta.end()) // pass save=... to save to a specific file
			save_weights(meta["save"]);
	}

protected:
	virtual void init_weights(const std::string& info){
		net.emplace_back(227812500); // create an empty weight table with size 15**6*4*5
		net.emplace_back(227812500); // now net.size() == 2; net[0].size() == 227812500; net[1].size() == 227812500
	}
	virtual void load_weights(const std::string& path){
		std::ifstream in(path, std::ios::in | std::ios::binary);
		if(!in.is_open()) std::exit(-1);
		uint32_t size;
		in.read(reinterpret_cast<char*>(&size), sizeof(size));
		net.resize(size);
		for(weight& w : net) in >> w;
		in.close();
	}
	virtual void save_weights(const std::string& path){
		std::ofstream out(path, std::ios::out | std::ios::binary | std::ios::trunc);
		if(!out.is_open()) std::exit(-1);
		uint32_t size = net.size();
		out.write(reinterpret_cast<char*>(&size), sizeof(size));
		for(weight& w : net) out << w;
		out.close();
	}

protected:
	std::vector<weight> net;
};

/**
 * base agent for agents with a learning rate
 */
class learning_agent : public weight_agent{
public:
	learning_agent(const std::string& args = "") : weight_agent(args), alpha(0.00001f/32){
		if(meta.find("alpha") != meta.end())
			alpha = float(meta["alpha"]);
	}
	virtual ~learning_agent() {}

protected:
	float alpha;
};

/**
 * random environment with bonus tile
 */
class rndenv : public random_agent{
public:
    rndenv(const std::string& args = "") : random_agent("name=random role=environment " + args),
		space({ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 }){}

    void reset(){
        bag = {1,1,1,1,2,2,2,2,3,3,3,3};
        std::shuffle(bag.begin(), bag.end(), engine);
        bonus.clear();
        bonus_max = 0;
        num_bag = 0;
        num_bonus = 0;
        hint = 0;
    }
    
    void generate_hint(const board& after){
        int new_hint;
        int b_max = after.max;
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
                float ratio = (num_bonus)/(num_bonus+num_bag);
                if(ratio <= 1/21){
                    std::uniform_int_distribution<> dis(0, bonus.size()-1);
                    new_hint = bonus[dis(gen)];
                    ++num_bonus;
                }
                else{
                    new_hint = bag.back();
                    bag.pop_back();
                    ++num_bag;
                }
            }
            //no bonus
            else{
                new_hint = bag.back();
                bag.pop_back();
                ++num_bag;
            }
        }
        //non-exist 48-tile
        else{
            new_hint = bag.back();
            bag.pop_back();
            ++num_bag;
        }
        hint = new_hint;
        return;
    }

	virtual action take_action(const board& after){
        int last = after.last;
        int h = hint;
        switch(last){
            case -1:
                if(last == -1){
                    std::shuffle(space.begin(), space.end(), engine);
                    if(h == 0){
                        h = bag.back();
                        bag.pop_back();
                        ++num_bag;
                    }
                    for (int pos : space){
                        if (after(pos) != 0) continue;
                        generate_hint(after);
                        return action::place(pos, h);
                    }
                }
            case 0:
                std::shuffle(op_0.begin(), op_0.end(), engine);
                for(int pos : op_0){
                    if(after(pos) != 0) continue;
                    generate_hint(after);
                    return action::place(pos, h);
                }
            case 1:
                std::shuffle(op_1.begin(), op_1.end(), engine);
                for(int pos : op_1){
                    if(after(pos) != 0) continue;
                    generate_hint(after);
                    return action::place(pos, h);
                }
            case 2:
                std::shuffle(op_2.begin(), op_2.end(), engine);
                for(int pos : op_2){
                    if(after(pos) != 0) continue;
                    generate_hint(after);
                    return action::place(pos, h);
                }
            case 3:
                std::shuffle(op_3.begin(), op_3.end(), engine);
                for(int pos : op_3){
                    if(after(pos) != 0) continue;
                    generate_hint(after);
                    return action::place(pos, h);
                }
            default:
                return action();
        }        
	}

private:
    std::random_device rd;
	std::array<int, 16> space;
    std::vector<int> bonus;
    std::array<int, 4> op;
    int bonus_max = 0;
    int num_bonus = 0;
    int num_bag = 0;
};

/**
 * td0 player
 * select a best action
 */
class player : public learning_agent{
public:
	player(const std::string& args = "") : learning_agent("name=learning role=player " + args){ std::srand(time(NULL)); }
 
    //get index according to the borad
    int find_index(int j, board as){
        int index = 0;
        int last = as.last;
        index = as.operator()(pattern[j][0]) + as.operator()(pattern[j][1]) * 15 + as.operator()(pattern[j][2]) * 225 + as.operator()(pattern[j][3]) * 3375 + as.operator()(pattern[j][4]) * 50625 + as.operator()(pattern[j][5]) * 759375;
        if(last == -1){ last = 4; }
        index = 11390625 * (4 * last + (as.hint - 1)) + index;//11390625 * 4 * last + 11390625 * (h - 1) + index;
        return index;
    }
    
    //search
    float search(board before, int k){
        int index;
        int layer = k-1;
        board after;
        //play node
        if(before.type == 'b'){
            float score = -99999;
            int r, v = 0;
            for(int i = 0; i < 4; ++i){                
                after = before;
                r = after.slide(i);
                if(r != -1){
                    v = 1;
                    after.type = 'a';
                    float child = search(after, layer);
                    child += r;
                    if(score < child) score = child;
                }
            }
            //existing child-node
            if(v == 1) return score;
            //non-existing child-node
            return -1;
        }
        //evil node
        else if(before.type == 'a'){
            float score = 0;            
            //depth = 0
            if(k == 0){
                for(int j = 0; j < 32; ++j){
                    index = find_index(j,before);
                    if(j < 16) score += net[0][index];
                    else score += net[1][index];
                }
                return score;
            }
            //depth != 0
            float value[3];
            float num_child = 0;
            int child[3];
            int last = before.last;            
            int v = 0;
            if(before.bag[0] == 0 && before.bag[1] == 0 && before.bag[2] == 0){
                before.bag[0] = 4;
                before.bag[1] = 4;
                before.bag[2] = 4;
            }
            child[0] = before.bag[0];
            child[1] = before.bag[1];
            child[2] = before.bag[2];
            if(last == 0){
                for(int pos : op_0){
                    if(before(pos) != 0) continue;
                    value[0] = 0;
                    value[1] = 0;
                    value[2] = 0;
                    after = before;
                    after.type = 'b';
                    if(before.hint != 4) after.place(pos,before.hint);
                    else after.place(pos, 4 + (std::rand()%((before.max-3) - 4 + 1)));                 
                    for(int i = 0; i < 3; ++i){
                        if(before.bag[i] > 0){//bag contains i
                            after.bag = before.bag;
                            //generate new hint
                            after.hint = i+1;
                            --after.bag[i];
                            float t = search(after, layer);
                            if(t != -1){
                                value[i] = t;
                                v = 1;
                                num_child += child[i];
                            }
                        }
                    }
                    score += value[0] * child[0] + value[1] * child[1] + value[2] * child[2];
                    //max >= 7(48)
                    if(before.max == 7){
                        after.bag = before.bag;
                        //generate new hint
                        after.hint = 4;
                        float t = search(after, layer);
                        if(t != -1){
                            score += t*0.05;
                            v = 1;
                            num_child += 0.05;
                        }
                    }
                }
            }
            else if(last == 1){
                for(int pos : op_1){
                    if(before(pos) != 0) continue;
                    value[0] = 0;
                    value[1] = 0;
                    value[2] = 0;
                    after = before;
                    after.type = 'b';
                    if(before.hint != 4) after.place(pos,before.hint);
                    else after.place(pos, 4 + (std::rand()%((before.max-3) - 4 + 1)));                 
                    for(int i = 0; i < 3; ++i){
                        if(before.bag[i] > 0){//bag contains i
                            after.bag = before.bag;
                            //generate new hint
                            after.hint = i+1;
                            --after.bag[i];
                            float t = search(after, layer);
                            if(t != -1){
                                value[i] = t;
                                v = 1;
                                num_child += child[i];
                            }
                        }
                    }
                    score += value[0] * child[0] + value[1] * child[1] + value[2] * child[2];
                    //max >= 7(48)
                    if(before.max == 7){
                        after.bag = before.bag;
                        //generate new hint
                        after.hint = 4;
                        float t = search(after, layer);
                        if(t != -1){
                            score += t*0.05;
                            v = 1;
                            num_child += 0.05;
                        }
                    }
                }
            }
            else if(last == 2){
                for(int pos : op_2){
                    if(before(pos) != 0) continue;
                    value[0] = 0;
                    value[1] = 0;
                    value[2] = 0;
                    after = before;
                    after.type = 'b';
                    if(before.hint != 4) after.place(pos,before.hint);
                    else after.place(pos, 4 + (std::rand()%((before.max-3) - 4 + 1)));                 
                    for(int i = 0; i < 3; ++i){
                        if(before.bag[i] > 0){//bag contains i
                            after.bag = before.bag;
                            //generate new hint
                            after.hint = i+1;
                            --after.bag[i];
                            float t = search(after, layer);
                            if(t != -1){
                                value[i] = t;
                                v = 1;
                                num_child += child[i];
                            }
                        }
                    }
                    score += value[0] * child[0] + value[1] * child[1] + value[2] * child[2];
                    //max >= 7(48)
                    if(before.max == 7){
                        after.bag = before.bag;
                        //generate new hint
                        after.hint = 4;
                        float t = search(after, layer);
                        if(t != -1){
                            score += t*0.05;
                            v = 1;
                            num_child += 0.05;
                        }
                    }
                }
            }
            else if(last == 3){
                for(int pos : op_3){
                    if(before(pos) != 0) continue;
                    value[0] = 0;
                    value[1] = 0;
                    value[2] = 0;
                    after = before;
                    after.type = 'b';
                    if(before.hint != 4) after.place(pos,before.hint);
                    else after.place(pos, 4 + (std::rand()%((before.max-3) - 4 + 1)));                 
                    for(int i = 0; i < 3; ++i){
                        if(before.bag[i] > 0){//bag contains i
                            after.bag = before.bag;
                            //generate new hint
                            after.hint = i+1;
                            --after.bag[i];
                            float t = search(after, layer);
                            if(t != -1){
                                value[i] = t;
                                v = 1;
                                num_child += child[i];
                            }
                        }
                    }
                    score += value[0] * child[0] + value[1] * child[1] + value[2] * child[2];
                    //max >= 7(48)
                    if(before.max == 7){
                        after.bag = before.bag;
                        //generate new hint
                        after.hint = 4;
                        float t = search(after, layer);
                        if(t != -1){
                            score += t*0.05;
                            v = 1;
                            num_child += 0.05;
                        }
                    }
                }
            }
            if(v == 1){
                score /= num_child;
                return score;
            }
            score = 0;
            for(int j = 0; j < 32; ++j){
                //std::cout<<before<<before.empty<<std::endl;
                index = find_index(j,before);                
                if(j < 16) score += net[0][index];
                else score += net[1][index];
            }
            return score;            
        }
        //std::cout<<"gg\n";
        return -1;
    }
    
    //action
	virtual action take_action(const board &before){
        int op, index, imdt_r;
        int valid = 0;
        float score = -999999;
        float current[4] = {0};
        std::array<int, 32> key;        
        board temp = before;
        temp.type = 'a';
        temp.hint = hint;
        temp.bag[0] = 0;
        temp.bag[1] = 0;
        temp.bag[2] = 0;
        for(std::vector<int>::iterator i = bag.begin(); i != bag.end(); ++i){
            if(*i == 1) ++temp.bag[0];
            else if(*i == 2) ++temp.bag[1];
            else if(*i == 3) ++temp.bag[2];
        }
        // find best action op
        /*#pragma omp parallel
        {
            int current_private[4] = {0};
            #pragma omp for
            for(int i = 0; i < 4; i++) {
              board as = temp;
              int reward = as.slide(i);
              current_private[i] = reward;
              if(reward != -1 ) { 
                  if(hint == 4) current_private[i] += search(as, 0); //searching i layers
                  else current_private[i] += search(as, 0);
              }
            }
            #pragma omp critical
            {
                for(int n = 0; n < 4; ++n) {
                  current[n] += current_private[n];
                }
            }
        }*/
        for(int i = 0; i < 4; ++i){
            board as = temp;
            int reward = as.slide(i);
            current[i] += reward;
            if(reward != -1){
                if(hint == 4 && as.max > 9) current[i] += search(as, 0);
                else current[i] += search(as, 2);//searching i layers
            }
        } 
        for(int i = 0; i < 4; ++i){
            if(current[i] != -1 && score < current[i]){
                score = current[i];
                op = i;
                valid = 1;
            }
        }
        //action found
        if(valid == 1){
            imdt_r = temp.slide(op);
            for(int j = 0; j < 32; ++j){
                index = find_index(j,temp);
                key[j] = index;
            }
            state_key.push_back(key);
            r.push_back(imdt_r);
            return action::slide(op);
        }
        //action not found
        return action();
    }
    
    //action
	/*virtual action take_action(const board &before) {
        int op = 0, valid = 0, index, imdt_r, reward;
        float t = -999999, direction_value;
        std::array<std::array<int, 32>, 4> key;
        board as;
        // find best action
        for(int i = 0; i < 4; ++i){
            direction_value = 0;
            as = before;
            reward = as.slide(i);//immediate reward
            if(reward != -1){
                valid = 1;
                direction_value += reward;
                for(int j = 0; j < 32; ++j){
                    index = find_index(j,as);
                    key[i][j] = index;
                    if(j < 16)
                        direction_value += net[0][index]; 
                    else
                        direction_value += net[1][index];
                }
                if(direction_value > t){
                    t = direction_value;
                    op = i;
                    imdt_r = reward;
                }
            }
        }
        //not terminal
        if(valid == 1){
            state_key.push_back(key[op]);
            r.push_back(imdt_r);
            return action::slide(op);
        }
        //terminal
        return action();
    }*/
    
    //training
    void training(){
        float sum = 0;
        float v_as, amend;
        std::vector<std::array<int, 32>>::reverse_iterator iter = state_key.rbegin();
        r.push_back(0);
        std::vector<int>::reverse_iterator rr = r.rbegin();
        while(iter != state_key.rend()){
            v_as = sum;
            sum = 0;
            for(int j = 0; j < 32; ++j){
                if(j<16)
                    sum += net[0][(*iter)[j]];
                else
                    sum += net[1][(*iter)[j]];
            }
            amend = alpha * ((*rr) + v_as - sum);
            sum = 0;
            for(int i = 0; i < 32; ++i){
                if(i < 16){
                    net[0][(*iter)[i]] += amend;
                    sum += net[0][(*iter)[i]];
                }
                else{
                    net[1][(*iter)[i]] += amend;
                    sum += net[1][(*iter)[i]];
                }
            }
            ++iter;
            ++rr;
        }
        r.clear();
        state_key.clear();
        return;
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
