// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow/core/framework/graph.proto

#ifndef PROTOBUF_tensorflow_2fcore_2fframework_2fgraph_2eproto__INCLUDED
#define PROTOBUF_tensorflow_2fcore_2fframework_2fgraph_2eproto__INCLUDED

#include <string>

#include <google/protobuf/stubs/common.h>

#if GOOGLE_PROTOBUF_VERSION < 3000000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please update
#error your headers.
#endif
#if 3000000 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/metadata.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/map.h>
#include <google/protobuf/map_field_inl.h>
#include <google/protobuf/unknown_field_set.h>
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
// @@protoc_insertion_point(includes)

namespace tensorflow {

// Internal implementation detail -- do not call these.
void protobuf_AddDesc_tensorflow_2fcore_2fframework_2fgraph_2eproto();
void protobuf_AssignDesc_tensorflow_2fcore_2fframework_2fgraph_2eproto();
void protobuf_ShutdownFile_tensorflow_2fcore_2fframework_2fgraph_2eproto();

class GraphDef;
class NodeDef;

// ===================================================================

class GraphDef : public ::google::protobuf::Message {
 public:
  GraphDef();
  virtual ~GraphDef();

  GraphDef(const GraphDef& from);

  inline GraphDef& operator=(const GraphDef& from) {
    CopyFrom(from);
    return *this;
  }

  static const ::google::protobuf::Descriptor* descriptor();
  static const GraphDef& default_instance();

  void Swap(GraphDef* other);

  // implements Message ----------------------------------------------

  inline GraphDef* New() const { return New(NULL); }

  GraphDef* New(::google::protobuf::Arena* arena) const;
  void CopyFrom(const ::google::protobuf::Message& from);
  void MergeFrom(const ::google::protobuf::Message& from);
  void CopyFrom(const GraphDef& from);
  void MergeFrom(const GraphDef& from);
  void Clear();
  bool IsInitialized() const;

  int ByteSize() const;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input);
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const;
  ::google::protobuf::uint8* SerializeWithCachedSizesToArray(::google::protobuf::uint8* output) const;
  int GetCachedSize() const { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const;
  void InternalSwap(GraphDef* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return _internal_metadata_.arena();
  }
  inline void* MaybeArenaPtr() const {
    return _internal_metadata_.raw_arena_ptr();
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // repeated .tensorflow.NodeDef node = 1;
  int node_size() const;
  void clear_node();
  static const int kNodeFieldNumber = 1;
  const ::tensorflow::NodeDef& node(int index) const;
  ::tensorflow::NodeDef* mutable_node(int index);
  ::tensorflow::NodeDef* add_node();
  ::google::protobuf::RepeatedPtrField< ::tensorflow::NodeDef >*
      mutable_node();
  const ::google::protobuf::RepeatedPtrField< ::tensorflow::NodeDef >&
      node() const;

  // optional .tensorflow.VersionDef versions = 4;
  bool has_versions() const;
  void clear_versions();
  static const int kVersionsFieldNumber = 4;
  const ::tensorflow::VersionDef& versions() const;
  ::tensorflow::VersionDef* mutable_versions();
  ::tensorflow::VersionDef* release_versions();
  void set_allocated_versions(::tensorflow::VersionDef* versions);

  // optional int32 version = 3 [deprecated = true];
  void clear_version() PROTOBUF_DEPRECATED;
  static const int kVersionFieldNumber = 3;
  ::google::protobuf::int32 version() const PROTOBUF_DEPRECATED;
  void set_version(::google::protobuf::int32 value) PROTOBUF_DEPRECATED;

  // optional .tensorflow.FunctionDefLibrary library = 2;
  bool has_library() const;
  void clear_library();
  static const int kLibraryFieldNumber = 2;
  const ::tensorflow::FunctionDefLibrary& library() const;
  ::tensorflow::FunctionDefLibrary* mutable_library();
  ::tensorflow::FunctionDefLibrary* release_library();
  void set_allocated_library(::tensorflow::FunctionDefLibrary* library);

  // @@protoc_insertion_point(class_scope:tensorflow.GraphDef)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  bool _is_default_instance_;
  ::google::protobuf::RepeatedPtrField< ::tensorflow::NodeDef > node_;
  ::tensorflow::VersionDef* versions_;
  ::tensorflow::FunctionDefLibrary* library_;
  ::google::protobuf::int32 version_;
  mutable int _cached_size_;
  friend void  protobuf_AddDesc_tensorflow_2fcore_2fframework_2fgraph_2eproto();
  friend void protobuf_AssignDesc_tensorflow_2fcore_2fframework_2fgraph_2eproto();
  friend void protobuf_ShutdownFile_tensorflow_2fcore_2fframework_2fgraph_2eproto();

  void InitAsDefaultInstance();
  static GraphDef* default_instance_;
};
// -------------------------------------------------------------------

class NodeDef : public ::google::protobuf::Message {
 public:
  NodeDef();
  virtual ~NodeDef();

  NodeDef(const NodeDef& from);

  inline NodeDef& operator=(const NodeDef& from) {
    CopyFrom(from);
    return *this;
  }

  static const ::google::protobuf::Descriptor* descriptor();
  static const NodeDef& default_instance();

  void Swap(NodeDef* other);

  // implements Message ----------------------------------------------

  inline NodeDef* New() const { return New(NULL); }

  NodeDef* New(::google::protobuf::Arena* arena) const;
  void CopyFrom(const ::google::protobuf::Message& from);
  void MergeFrom(const ::google::protobuf::Message& from);
  void CopyFrom(const NodeDef& from);
  void MergeFrom(const NodeDef& from);
  void Clear();
  bool IsInitialized() const;

  int ByteSize() const;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input);
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const;
  ::google::protobuf::uint8* SerializeWithCachedSizesToArray(::google::protobuf::uint8* output) const;
  int GetCachedSize() const { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const;
  void InternalSwap(NodeDef* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return _internal_metadata_.arena();
  }
  inline void* MaybeArenaPtr() const {
    return _internal_metadata_.raw_arena_ptr();
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const;

  // nested types ----------------------------------------------------


  // accessors -------------------------------------------------------

  // optional string name = 1;
  void clear_name();
  static const int kNameFieldNumber = 1;
  const ::std::string& name() const;
  void set_name(const ::std::string& value);
  void set_name(const char* value);
  void set_name(const char* value, size_t size);
  ::std::string* mutable_name();
  ::std::string* release_name();
  void set_allocated_name(::std::string* name);

  // optional string op = 2;
  void clear_op();
  static const int kOpFieldNumber = 2;
  const ::std::string& op() const;
  void set_op(const ::std::string& value);
  void set_op(const char* value);
  void set_op(const char* value, size_t size);
  ::std::string* mutable_op();
  ::std::string* release_op();
  void set_allocated_op(::std::string* op);

  // repeated string input = 3;
  int input_size() const;
  void clear_input();
  static const int kInputFieldNumber = 3;
  const ::std::string& input(int index) const;
  ::std::string* mutable_input(int index);
  void set_input(int index, const ::std::string& value);
  void set_input(int index, const char* value);
  void set_input(int index, const char* value, size_t size);
  ::std::string* add_input();
  void add_input(const ::std::string& value);
  void add_input(const char* value);
  void add_input(const char* value, size_t size);
  const ::google::protobuf::RepeatedPtrField< ::std::string>& input() const;
  ::google::protobuf::RepeatedPtrField< ::std::string>* mutable_input();

  // optional string device = 4;
  void clear_device();
  static const int kDeviceFieldNumber = 4;
  const ::std::string& device() const;
  void set_device(const ::std::string& value);
  void set_device(const char* value);
  void set_device(const char* value, size_t size);
  ::std::string* mutable_device();
  ::std::string* release_device();
  void set_allocated_device(::std::string* device);

  // map<string, .tensorflow.AttrValue> attr = 5;
  int attr_size() const;
  void clear_attr();
  static const int kAttrFieldNumber = 5;
  const ::google::protobuf::Map< ::std::string, ::tensorflow::AttrValue >&
      attr() const;
  ::google::protobuf::Map< ::std::string, ::tensorflow::AttrValue >*
      mutable_attr();

  // @@protoc_insertion_point(class_scope:tensorflow.NodeDef)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  bool _is_default_instance_;
  ::google::protobuf::internal::ArenaStringPtr name_;
  ::google::protobuf::internal::ArenaStringPtr op_;
  ::google::protobuf::RepeatedPtrField< ::std::string> input_;
  ::google::protobuf::internal::ArenaStringPtr device_;
  typedef ::google::protobuf::internal::MapEntryLite<
      ::std::string, ::tensorflow::AttrValue,
      ::google::protobuf::internal::WireFormatLite::TYPE_STRING,
      ::google::protobuf::internal::WireFormatLite::TYPE_MESSAGE,
      0 >
      NodeDef_AttrEntry;
  ::google::protobuf::internal::MapField<
      ::std::string, ::tensorflow::AttrValue,
      ::google::protobuf::internal::WireFormatLite::TYPE_STRING,
      ::google::protobuf::internal::WireFormatLite::TYPE_MESSAGE,
      0 > attr_;
  mutable int _cached_size_;
  friend void  protobuf_AddDesc_tensorflow_2fcore_2fframework_2fgraph_2eproto();
  friend void protobuf_AssignDesc_tensorflow_2fcore_2fframework_2fgraph_2eproto();
  friend void protobuf_ShutdownFile_tensorflow_2fcore_2fframework_2fgraph_2eproto();

  void InitAsDefaultInstance();
  static NodeDef* default_instance_;
};
// ===================================================================


// ===================================================================

#if !PROTOBUF_INLINE_NOT_IN_HEADERS
// GraphDef

// repeated .tensorflow.NodeDef node = 1;
inline int GraphDef::node_size() const {
  return node_.size();
}
inline void GraphDef::clear_node() {
  node_.Clear();
}
inline const ::tensorflow::NodeDef& GraphDef::node(int index) const {
  // @@protoc_insertion_point(field_get:tensorflow.GraphDef.node)
  return node_.Get(index);
}
inline ::tensorflow::NodeDef* GraphDef::mutable_node(int index) {
  // @@protoc_insertion_point(field_mutable:tensorflow.GraphDef.node)
  return node_.Mutable(index);
}
inline ::tensorflow::NodeDef* GraphDef::add_node() {
  // @@protoc_insertion_point(field_add:tensorflow.GraphDef.node)
  return node_.Add();
}
inline ::google::protobuf::RepeatedPtrField< ::tensorflow::NodeDef >*
GraphDef::mutable_node() {
  // @@protoc_insertion_point(field_mutable_list:tensorflow.GraphDef.node)
  return &node_;
}
inline const ::google::protobuf::RepeatedPtrField< ::tensorflow::NodeDef >&
GraphDef::node() const {
  // @@protoc_insertion_point(field_list:tensorflow.GraphDef.node)
  return node_;
}

// optional .tensorflow.VersionDef versions = 4;
inline bool GraphDef::has_versions() const {
  return !_is_default_instance_ && versions_ != NULL;
}
inline void GraphDef::clear_versions() {
  if (GetArenaNoVirtual() == NULL && versions_ != NULL) delete versions_;
  versions_ = NULL;
}
inline const ::tensorflow::VersionDef& GraphDef::versions() const {
  // @@protoc_insertion_point(field_get:tensorflow.GraphDef.versions)
  return versions_ != NULL ? *versions_ : *default_instance_->versions_;
}
inline ::tensorflow::VersionDef* GraphDef::mutable_versions() {
  
  if (versions_ == NULL) {
    versions_ = new ::tensorflow::VersionDef;
  }
  // @@protoc_insertion_point(field_mutable:tensorflow.GraphDef.versions)
  return versions_;
}
inline ::tensorflow::VersionDef* GraphDef::release_versions() {
  
  ::tensorflow::VersionDef* temp = versions_;
  versions_ = NULL;
  return temp;
}
inline void GraphDef::set_allocated_versions(::tensorflow::VersionDef* versions) {
  delete versions_;
  versions_ = versions;
  if (versions) {
    
  } else {
    
  }
  // @@protoc_insertion_point(field_set_allocated:tensorflow.GraphDef.versions)
}

// optional int32 version = 3 [deprecated = true];
inline void GraphDef::clear_version() {
  version_ = 0;
}
inline ::google::protobuf::int32 GraphDef::version() const {
  // @@protoc_insertion_point(field_get:tensorflow.GraphDef.version)
  return version_;
}
inline void GraphDef::set_version(::google::protobuf::int32 value) {
  
  version_ = value;
  // @@protoc_insertion_point(field_set:tensorflow.GraphDef.version)
}

// optional .tensorflow.FunctionDefLibrary library = 2;
inline bool GraphDef::has_library() const {
  return !_is_default_instance_ && library_ != NULL;
}
inline void GraphDef::clear_library() {
  if (GetArenaNoVirtual() == NULL && library_ != NULL) delete library_;
  library_ = NULL;
}
inline const ::tensorflow::FunctionDefLibrary& GraphDef::library() const {
  // @@protoc_insertion_point(field_get:tensorflow.GraphDef.library)
  return library_ != NULL ? *library_ : *default_instance_->library_;
}
inline ::tensorflow::FunctionDefLibrary* GraphDef::mutable_library() {
  
  if (library_ == NULL) {
    library_ = new ::tensorflow::FunctionDefLibrary;
  }
  // @@protoc_insertion_point(field_mutable:tensorflow.GraphDef.library)
  return library_;
}
inline ::tensorflow::FunctionDefLibrary* GraphDef::release_library() {
  
  ::tensorflow::FunctionDefLibrary* temp = library_;
  library_ = NULL;
  return temp;
}
inline void GraphDef::set_allocated_library(::tensorflow::FunctionDefLibrary* library) {
  delete library_;
  library_ = library;
  if (library) {
    
  } else {
    
  }
  // @@protoc_insertion_point(field_set_allocated:tensorflow.GraphDef.library)
}

// -------------------------------------------------------------------

// NodeDef

// optional string name = 1;
inline void NodeDef::clear_name() {
  name_.ClearToEmptyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline const ::std::string& NodeDef::name() const {
  // @@protoc_insertion_point(field_get:tensorflow.NodeDef.name)
  return name_.GetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline void NodeDef::set_name(const ::std::string& value) {
  
  name_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), value);
  // @@protoc_insertion_point(field_set:tensorflow.NodeDef.name)
}
inline void NodeDef::set_name(const char* value) {
  
  name_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::string(value));
  // @@protoc_insertion_point(field_set_char:tensorflow.NodeDef.name)
}
inline void NodeDef::set_name(const char* value, size_t size) {
  
  name_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(),
      ::std::string(reinterpret_cast<const char*>(value), size));
  // @@protoc_insertion_point(field_set_pointer:tensorflow.NodeDef.name)
}
inline ::std::string* NodeDef::mutable_name() {
  
  // @@protoc_insertion_point(field_mutable:tensorflow.NodeDef.name)
  return name_.MutableNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline ::std::string* NodeDef::release_name() {
  
  return name_.ReleaseNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline void NodeDef::set_allocated_name(::std::string* name) {
  if (name != NULL) {
    
  } else {
    
  }
  name_.SetAllocatedNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), name);
  // @@protoc_insertion_point(field_set_allocated:tensorflow.NodeDef.name)
}

// optional string op = 2;
inline void NodeDef::clear_op() {
  op_.ClearToEmptyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline const ::std::string& NodeDef::op() const {
  // @@protoc_insertion_point(field_get:tensorflow.NodeDef.op)
  return op_.GetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline void NodeDef::set_op(const ::std::string& value) {
  
  op_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), value);
  // @@protoc_insertion_point(field_set:tensorflow.NodeDef.op)
}
inline void NodeDef::set_op(const char* value) {
  
  op_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::string(value));
  // @@protoc_insertion_point(field_set_char:tensorflow.NodeDef.op)
}
inline void NodeDef::set_op(const char* value, size_t size) {
  
  op_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(),
      ::std::string(reinterpret_cast<const char*>(value), size));
  // @@protoc_insertion_point(field_set_pointer:tensorflow.NodeDef.op)
}
inline ::std::string* NodeDef::mutable_op() {
  
  // @@protoc_insertion_point(field_mutable:tensorflow.NodeDef.op)
  return op_.MutableNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline ::std::string* NodeDef::release_op() {
  
  return op_.ReleaseNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline void NodeDef::set_allocated_op(::std::string* op) {
  if (op != NULL) {
    
  } else {
    
  }
  op_.SetAllocatedNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), op);
  // @@protoc_insertion_point(field_set_allocated:tensorflow.NodeDef.op)
}

// repeated string input = 3;
inline int NodeDef::input_size() const {
  return input_.size();
}
inline void NodeDef::clear_input() {
  input_.Clear();
}
inline const ::std::string& NodeDef::input(int index) const {
  // @@protoc_insertion_point(field_get:tensorflow.NodeDef.input)
  return input_.Get(index);
}
inline ::std::string* NodeDef::mutable_input(int index) {
  // @@protoc_insertion_point(field_mutable:tensorflow.NodeDef.input)
  return input_.Mutable(index);
}
inline void NodeDef::set_input(int index, const ::std::string& value) {
  // @@protoc_insertion_point(field_set:tensorflow.NodeDef.input)
  input_.Mutable(index)->assign(value);
}
inline void NodeDef::set_input(int index, const char* value) {
  input_.Mutable(index)->assign(value);
  // @@protoc_insertion_point(field_set_char:tensorflow.NodeDef.input)
}
inline void NodeDef::set_input(int index, const char* value, size_t size) {
  input_.Mutable(index)->assign(
    reinterpret_cast<const char*>(value), size);
  // @@protoc_insertion_point(field_set_pointer:tensorflow.NodeDef.input)
}
inline ::std::string* NodeDef::add_input() {
  return input_.Add();
}
inline void NodeDef::add_input(const ::std::string& value) {
  input_.Add()->assign(value);
  // @@protoc_insertion_point(field_add:tensorflow.NodeDef.input)
}
inline void NodeDef::add_input(const char* value) {
  input_.Add()->assign(value);
  // @@protoc_insertion_point(field_add_char:tensorflow.NodeDef.input)
}
inline void NodeDef::add_input(const char* value, size_t size) {
  input_.Add()->assign(reinterpret_cast<const char*>(value), size);
  // @@protoc_insertion_point(field_add_pointer:tensorflow.NodeDef.input)
}
inline const ::google::protobuf::RepeatedPtrField< ::std::string>&
NodeDef::input() const {
  // @@protoc_insertion_point(field_list:tensorflow.NodeDef.input)
  return input_;
}
inline ::google::protobuf::RepeatedPtrField< ::std::string>*
NodeDef::mutable_input() {
  // @@protoc_insertion_point(field_mutable_list:tensorflow.NodeDef.input)
  return &input_;
}

// optional string device = 4;
inline void NodeDef::clear_device() {
  device_.ClearToEmptyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline const ::std::string& NodeDef::device() const {
  // @@protoc_insertion_point(field_get:tensorflow.NodeDef.device)
  return device_.GetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline void NodeDef::set_device(const ::std::string& value) {
  
  device_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), value);
  // @@protoc_insertion_point(field_set:tensorflow.NodeDef.device)
}
inline void NodeDef::set_device(const char* value) {
  
  device_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::string(value));
  // @@protoc_insertion_point(field_set_char:tensorflow.NodeDef.device)
}
inline void NodeDef::set_device(const char* value, size_t size) {
  
  device_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(),
      ::std::string(reinterpret_cast<const char*>(value), size));
  // @@protoc_insertion_point(field_set_pointer:tensorflow.NodeDef.device)
}
inline ::std::string* NodeDef::mutable_device() {
  
  // @@protoc_insertion_point(field_mutable:tensorflow.NodeDef.device)
  return device_.MutableNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline ::std::string* NodeDef::release_device() {
  
  return device_.ReleaseNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline void NodeDef::set_allocated_device(::std::string* device) {
  if (device != NULL) {
    
  } else {
    
  }
  device_.SetAllocatedNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), device);
  // @@protoc_insertion_point(field_set_allocated:tensorflow.NodeDef.device)
}

// map<string, .tensorflow.AttrValue> attr = 5;
inline int NodeDef::attr_size() const {
  return attr_.size();
}
inline void NodeDef::clear_attr() {
  attr_.Clear();
}
inline const ::google::protobuf::Map< ::std::string, ::tensorflow::AttrValue >&
NodeDef::attr() const {
  // @@protoc_insertion_point(field_map:tensorflow.NodeDef.attr)
  return attr_.GetMap();
}
inline ::google::protobuf::Map< ::std::string, ::tensorflow::AttrValue >*
NodeDef::mutable_attr() {
  // @@protoc_insertion_point(field_mutable_map:tensorflow.NodeDef.attr)
  return attr_.MutableMap();
}

#endif  // !PROTOBUF_INLINE_NOT_IN_HEADERS
// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)

}  // namespace tensorflow

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_tensorflow_2fcore_2fframework_2fgraph_2eproto__INCLUDED
