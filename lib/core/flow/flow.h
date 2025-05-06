/*
 * Copyright (c) 2017-2024 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of Nvidia Corporation and its affiliates
 * (the "Company") and all right, title, and interest in and to the software
 * product, including all associated intellectual property rights, are and
 * shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 */

#ifndef RMAX_APPS_LIB_LIB_CORE_FLOW_FLOW_H_

#include <cstddef>
#include <string>
#include <tuple>

/* Platform specific headers and declarations */
#if defined(_WIN32)
#include <ws2tcpip.h>
#pragma comment(lib, "Ws2_32.lib")
#elif defined(__linux__)
#include <arpa/inet.h>
#endif

#include "services/error_handling/return_status.h"

using namespace ral::lib::services;

namespace ral
{
namespace lib
{
namespace core
{

/**
 * @brief: Flow interface.
 *
 * This interfaces represents network flow.
 */
class IFlow
{
private:
    size_t m_id;
public:
    /**
     * @brief: Flow constructor.
     *
     * @param [in] id: Flow ID.
     */
    IFlow(size_t id);
    virtual ~IFlow() = default;
    /**
     * @brief: Returns flow ID.
     *
     * @returns: Flow ID.
     */
    size_t get_id() const { return m_id; }
    /**
     * @brief: Sets socket address.
     *
     * The method sets IP and port to sockaddr_in type address.
     *
     * @param [in] ip: Address IP.
     * @param [in] port: Address port.
     * @param [out] address: Address reference to set.
     *
     * @returns: Status of the operation.
     */
    static ReturnStatus set_socket_address(const std::string& ip, uint16_t port, sockaddr_in& address);
};
/**
 * @brief: Two tuple flow interface.
 *
 * This interfaces represents 2 tuple network flow.
 * It can be used to represent send/receive flows.
 */
class TwoTupleFlow : public IFlow
{
template <typename T> friend struct std::hash;
private:
    std::string m_ip;
    uint16_t m_port;
    sockaddr_in m_address;
public:
    /**
     * @brief: TwoTupleFlow constructor.
     *
     * @param [in] id: ID.
     * @param [in] ip: IP address.
     * @param [in] port: Port number.
     */
    TwoTupleFlow(size_t id, const std::string& ip, uint16_t port);
    virtual ~TwoTupleFlow() = default;
    /**
     * @brief: Returns network flow IP.
     *
     * @returns: Network flow IP string.
     */
    const std::string& get_ip() const { return m_ip; }
    /**
     * @brief: Returns network flow port.
     *
     * @returns: Network flow port number.
     */
    uint16_t get_port() const { return m_port; }
    /**
     * @brief: Returns socket address representation of the flow.
     *
     * @returns: Socket address represents the flow.
     */
    sockaddr& get_socket_address() { return reinterpret_cast<sockaddr&>(m_address); }
    const sockaddr& get_socket_address() const { return reinterpret_cast<const sockaddr&>(m_address); }
    /**
     * @brief: Equality operator.
     *
     * @return: true if operands are equal.
     */
    bool operator==(const TwoTupleFlow& rhs) const noexcept
    {
        return std::tie(m_ip, m_port) == std::tie(rhs.m_ip, rhs.m_port);
    }
};
/**
 * @brief: Four tuple flow interface.
 *
 * This interfaces represents 4 tuple network flow.
 */
class FourTupleFlow : public IFlow
{
template <typename T> friend struct std::hash;
private:
    TwoTupleFlow m_source_flow;
    TwoTupleFlow m_destination_flow;

public:
    /**
     * @brief: FourTupleFlow constructor.
     *
     * @param [in] id: ID.
     * @param [in] source_ip: Source IP address.
     * @param [in] source_port: Source port number.
     * @param [in] destination_ip: Destination IP address.
     * @param [in] destination_port: Destination port number.
     */
    FourTupleFlow(
        size_t id, const std::string& source_ip, uint16_t source_port,
        const std::string& destination_ip, uint16_t destination_port);
    /**
     * @brief: FourTupleFlow constructor.
     *
     * @param [in] id: ID.
     * @param [in] source_flow: Source flow of the 4 tuple.
     * @param [in] destination_flow: Destination flow of the 4 tuple.
     */
    FourTupleFlow(size_t id, const TwoTupleFlow& source_flow, const TwoTupleFlow& destination_flow);
    virtual ~FourTupleFlow() = default;
    /**
     * @brief: Returns source IP.
     *
     * @returns: Source flow IP string.
     */
    const std::string& get_source_ip() const { return m_source_flow.get_ip(); }
    /**
     * @brief: Returns source port.
     *
     * @returns: Source flow port number.
     */
    uint16_t get_source_port() const { return m_source_flow.get_port(); };
    /**
     * @brief: Returns destination IP.
     *
     * @returns: Destination flow IP string.
     */
    const std::string& get_destination_ip() const { return m_destination_flow.get_ip(); }
    /**
     * @brief: Returns destination port.
     *
     * @returns: Destination flow port number.
     */
    uint16_t get_destination_port() const { return m_destination_flow.get_port(); };
    /**
     * @brief: Returns source flow of the 4 tuple.
     *
     * @returns: Source flow.
     */
    const TwoTupleFlow& get_source_flow() const { return m_source_flow; }
    /**
     * @brief: Returns destination flow of the 4 tuple.
     *
     * @returns: Destination flow.
     */
    const TwoTupleFlow& get_destination_flow() const { return m_destination_flow; }
    /**
     * @brief: Returns socket address representation of the source flow.
     *
     * @returns: Socket address represents source flow.
     */
    sockaddr& get_source_socket_address() { return m_source_flow.get_socket_address(); };
    const sockaddr& get_source_socket_address() const { return m_source_flow.get_socket_address(); };
    /**
     * @brief: Returns socket address representation of the destination flow.
     *
     * @returns: Socket address represents destination flow.
     */
    sockaddr& get_destination_socket_address() { return m_destination_flow.get_socket_address(); };
    const sockaddr& get_destination_socket_address() const { return m_destination_flow.get_socket_address(); };
    /**
     * @brief: Equality operator.
     *
     * @return: true if operands are equal.
     */
    bool operator==(const FourTupleFlow& rhs) const noexcept
    {
        return std::tie(m_source_flow, m_destination_flow) == std::tie(rhs.m_source_flow, rhs.m_destination_flow);
    }
};

} // namespace core
} // namespace lib
} // namespace ral

namespace std {
template<>
struct hash<ral::lib::core::TwoTupleFlow>
{
    /**
     * @brief: Hash function for TwoTupleFlow.
     *
     * @return: Hash code.
     */
    std::size_t operator()(const ral::lib::core::TwoTupleFlow& s) const noexcept
    {
        std::size_t h1 = std::hash<std::string>{}(s.m_ip);
        std::size_t h2 = std::hash<uint16_t>{}(s.m_port);
        return h1 ^ (h2 << 1);
    }
};

template<>
struct hash<ral::lib::core::FourTupleFlow>
{
    /**
     * @brief: Hash function for FourTupleFlow.
     *
     * @return: Hash code.
     */
    std::size_t operator()(const ral::lib::core::FourTupleFlow& s) const noexcept
    {
        std::size_t h1 = std::hash<ral::lib::core::TwoTupleFlow>{}(s.m_source_flow);
        std::size_t h2 = std::hash<ral::lib::core::TwoTupleFlow>{}(s.m_destination_flow);
        return h1 ^ (h2 << 1);
    }
};
} // namespace std

#define RMAX_APPS_LIB_LIB_CORE_FLOW_FLOW_H_
#endif /* RMAX_APPS_LIB_LIB_CORE_FLOW_FLOW_H_ */
