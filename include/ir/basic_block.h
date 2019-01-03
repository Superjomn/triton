#ifndef TDL_INCLUDE_IR_BASIC_BLOCK_H
#define TDL_INCLUDE_IR_BASIC_BLOCK_H

#include <string>
#include <list>
#include "value.h"

namespace tdl{
namespace ir{

class context;
class function;
class instruction;

/* Basic Block */
class basic_block: public value{
public:
  // instruction iterator types
  typedef std::list<instruction*>                inst_list_t;
  typedef inst_list_t::iterator                  iterator;
  typedef inst_list_t::const_iterator            const_iterator;
  typedef inst_list_t::reverse_iterator          reverse_iterator;
  typedef inst_list_t::const_reverse_iterator    const_reverse_iterator;


public:
  // accessors
  function* get_parent() { return parent_; }
  context& get_context() { return ctx_; }

  // get iterator to first instruction that is not a phi
  iterator get_first_non_phi();

  // get instruction list
  inst_list_t           &get_inst_list()       { return inst_list_; }

  // instruction iterator functions
  inline iterator                begin()       { return inst_list_.begin(); }
  inline const_iterator          begin() const { return inst_list_.begin(); }
  inline iterator                end  ()       { return inst_list_.end();   }
  inline const_iterator          end  () const { return inst_list_.end();   }

  inline reverse_iterator        rbegin()       { return inst_list_.rbegin(); }
  inline const_reverse_iterator  rbegin() const { return inst_list_.rbegin(); }
  inline reverse_iterator        rend  ()       { return inst_list_.rend();   }
  inline const_reverse_iterator  rend  () const { return inst_list_.rend();   }

  inline size_t                   size() const { return inst_list_.size();  }
  inline bool                    empty() const { return inst_list_.empty(); }
  inline const instruction      &front() const { return *inst_list_.front(); }
  inline       instruction      &front()       { return *inst_list_.front(); }
  inline const instruction       &back() const { return *inst_list_.back();  }
  inline       instruction       &back()       { return *inst_list_.back();  }

  // predecessors
  const std::vector<basic_block*>& get_predecessors() const { return preds_; }
  void add_predecessor(basic_block* pred);

  // factory functions
  static basic_block* create(context &ctx, const std::string &name, function *parent);

private:
  context &ctx_;
  std::string name_;
  function *parent_;
  std::vector<basic_block*> preds_;
  inst_list_t inst_list_;
};

}
}

#endif
