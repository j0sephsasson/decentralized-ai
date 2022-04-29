// SPDX-License-Identifier: MIT
pragma solidity >=0.6.x;

import "../node_modules/openzeppelin-solidity/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/utils/Counters.sol";

contract DojoApp is ERC721 {
    using Counters for Counters.Counter;
    Counters.Counter private _tokenIds;
    
    struct Model {
        string name;
        uint256 accuracy;
        string description;
        uint256 tokenID;
    }

    constructor() ERC721("DojoApp", "AI")  { }

    mapping(uint256 => Model) public tokenIdToModel;
    mapping(uint256 => uint256) public modelsForSale;
    mapping(string => Model) public nameToModel;


    // Function that allows you to convert an address into a payable address
    function _make_payable(address x) internal pure returns(address payable) {
        return payable(x);
    }

    // Create Model using the Struct
    function createModel(string memory _name, uint256 _accuracy, string memory _description) public payable { // Passing the name and tokenId as a parameters
        require(nameToModel[_name].accuracy == 0, "Model with the same name already exists");

        _tokenIds.increment();
        uint256 newItemId = _tokenIds.current();

        Model memory newModel = Model(_name, _accuracy, _description, newItemId); // Model is an struct so we are creating a new Model
        nameToModel[_name] = newModel; // Creating in memory the name -> Model mapping
        tokenIdToModel[newItemId] = newModel; // Creating in memory the Model -> tokenId mapping
        _mint(msg.sender, newItemId); // _mint assign the the Model with _tokenId to the sender address (ownership)

        address platformWallet = 0xa8e7Db9573a0b23875b028a9Ebb45D7B2E6a2F84;
        address payable platformWalletPayable = _make_payable(platformWallet);

        platformWalletPayable.transfer(msg.value);
    }

    // Putting an Model for sale (Adding the Model tokenid into the mapping modelsForSale, first verify that the sender is the owner)
    function putModelUpForSale(string memory _name, uint256 _price) public {
        require(nameToModel[_name].tokenID != 0, "You can't sell a Model that doesn't exist!");
        require(ownerOf(nameToModel[_name].tokenID) == msg.sender, "You can't sell a Model you don't own!");
        modelsForSale[nameToModel[_name].tokenID] = _price;
    }

    function buyModel(string memory _name) public  payable {
        require(modelsForSale[nameToModel[_name].tokenID] > 0, "The Model should be up for sale");
        uint256 ModelCost = modelsForSale[nameToModel[_name].tokenID];
        address ownerAddress = ownerOf(nameToModel[_name].tokenID);
        require(msg.value > ModelCost, "You need to have enough Ether");
        _transfer(ownerAddress, msg.sender, nameToModel[_name].tokenID); 
        address payable ownerAddressPayable = _make_payable(ownerAddress); // We need to make this conversion to be able to use transfer() function to transfer ethers
        ownerAddressPayable.transfer(ModelCost);
        address platformWallet = 0xa8e7Db9573a0b23875b028a9Ebb45D7B2E6a2F84;
        address payable platformWalletPayable = _make_payable(platformWallet);
        
        platformWalletPayable.transfer(msg.value - ModelCost);
       
        delete modelsForSale[nameToModel[_name].tokenID];
    }

    function checkPrice(string memory _name) public view returns(uint256) {
        uint256 ModelCost = modelsForSale[nameToModel[_name].tokenID];
        return ModelCost;
    }

}